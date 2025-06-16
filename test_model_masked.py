import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from .dataset import MultimodalDataset
from .model import FusionModel
from .loss import PMRLoss
from .config import Config
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from .llama_integration import LlamaIntegration
from .bert_integration import BertIntegration
import warnings
from tabulate import tabulate

warnings.filterwarnings("ignore",
                        message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn.functional as F
from .feature_bank import FeatureBank

label_map = {
    0: "Angry",
    1: "Bored",
    2: "Calm",
    3: "Fear",
    4: "Happy",
    5: "Love",
    6: "Sad"
}
llama_integration = None
bert_integration = None


def get_llama():
    global llama_integration
    if llama_integration is None:
        print("Initializing LLaMA model...")
        llama_integration = LlamaIntegration()
    return llama_integration


def get_bert():
    global bert_integration
    if bert_integration is None:
        print("Initializing BERT model...")
        classnames = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
        bert_integration = BertIntegration(classnames=classnames)
    return bert_integration


def train_fusion_model(model, train_loader, val_loaders, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    best_val_acc = 0
    best_epoch = 0

    history = []

    print("\n=== Feature Bank Initialization ===")
    print("Initializing feature bank object...")
    model.feature_bank = FeatureBank(device)
    model.feature_bank.build_bank(model, train_loader)
    print("=== Feature Bank Initialization Completed ===\n")

    headers = ['Epoch', 'Train Loss', 'Dataset',
               'Fusion Acc', 'BERT Acc',
               'Fusion F1 Macro', 'BERT F1 Macro',
               'Fusion F1 Weighted', 'BERT F1 Weighted']

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_steps = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch in train_bar:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, prototypes, features, similarities, text_features, explanations, _ = model(images, texts)
            loss, _, _ = criterion(logits, prototypes, features, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_steps

        results_table = []
        for name, val_loader_tuple in val_loaders.items():
            if isinstance(val_loader_tuple, tuple):
                val_loader, specific_model = val_loader_tuple
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val: {name}]')
                fusion_acc, fusion_f1_macro, fusion_f1_weighted, \
                bert_acc, bert_f1_macro, bert_f1_weighted = validate(
                    specific_model, val_bar, criterion, device, epoch + 1
                )
            else:
                val_loader = val_loader_tuple
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val: {name}]')
                fusion_acc, fusion_f1_macro, fusion_f1_weighted, \
                bert_acc, bert_f1_macro, bert_f1_weighted = validate(
                    model, val_bar, criterion, device, epoch + 1
                )

            epoch_results = [
                epoch + 1,
                f"{avg_train_loss:.4f}",
                name,
                f"{fusion_acc:.4f}",
                f"{bert_acc:.4f}" if bert_acc is not None else "N/A",
                f"{fusion_f1_macro:.4f}",
                f"{bert_f1_macro:.4f}" if bert_f1_macro is not None else "N/A",
                f"{fusion_f1_weighted:.4f}",
                f"{bert_f1_weighted:.4f}" if bert_f1_weighted is not None else "N/A"
            ]

            results_table.append(epoch_results)
            history.append(epoch_results)

        print("\nEpoch Results:")
        print(tabulate(results_table, headers=headers, tablefmt="grid"))

        if scheduler is not None:
            scheduler.step()

        current_val_acc = fusion_acc
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch + 1
            print(f'\nNew best model saved! (Epoch {best_epoch}, Accuracy: {best_val_acc:.4f})')
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}')

    print("\nComplete Training History:")
    print(tabulate(history, headers=headers, tablefmt="grid",
                   floatfmt=".4f", numalign="right"))

    best_results = history[best_epoch - 1]
    print("\nBest Model Performance (Epoch {}):".format(best_epoch))
    print(f"Fusion Accuracy: {best_results[3]}")
    print(f"Fusion F1 Macro: {best_results[5]}")
    print(f"Fusion F1 Weighted: {best_results[7]}")
    if best_results[4] != "N/A":
        print(f"BERT Accuracy: {best_results[4]}")
        print(f"BERT F1 Macro: {best_results[6]}")
        print(f"BERT F1 Weighted: {best_results[8]}")

    return best_val_acc, best_epoch


def construct_fusion_output(logits, similarities, explanations, reference_info=None):
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)

    output = {
        'predictions': predictions.cpu().numpy(),
        'probabilities': probs.detach().cpu().numpy(),
        'similarities': similarities.detach().cpu().numpy(),
        'explanations': explanations
    }

    if reference_info is not None:
        output['reference_samples'] = reference_info['samples']
        output['reference_similarities'] = reference_info['similarities'].cpu().numpy()

    return output


def validate(model, val_loader, criterion, device, epoch=None):
    model.eval()
    fusion_predictions = []
    fusion_targets = []
    bert_predictions = []
    bert_targets = []
    bert = get_bert()

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            batch_size = len(texts)

            outputs = model.extract_features(images, texts)
            if outputs is None:
                continue

            logits, prototypes, features, similarities, text_features, explanations, modality_weights, image_captions = outputs
            predictions = torch.argmax(logits, dim=1)

            fusion_predictions.extend(predictions.cpu().numpy().tolist())
            fusion_targets.extend(labels.cpu().numpy().tolist())

            try:
                bert_preds = []
                for i in range(batch_size):
                    true_label = label_map[labels[i].item()]
                    fusion_pred = label_map[predictions[i].item()]

                    fusion_output = {
                        'text': {'content': texts[i]},
                        'image': {'description': image_captions[i]},
                        'explanation': explanations[i] if explanations is not None else None
                    }

                    if features is not None and hasattr(model, 'feature_bank'):
                        text_importance = torch.norm(text_features[i]).item()
                        image_importance = torch.norm(features[i][features[i].size(0) // 2:]).item()
                        total = text_importance + image_importance
                        modality_weights = (text_importance / total, image_importance / total)

                        reference_samples, ref_similarities = model.feature_bank.retrieve_diverse_samples(
                            features[i].unsqueeze(0),
                            modality_weights=modality_weights
                        )
                        result = bert.process_fusion_output(fusion_output, reference_samples, ref_similarities)
                    else:
                        result = bert.process_fusion_output(fusion_output)

                    bert_preds.append(result['sentiment'])
                    bert_output = label_map[result['sentiment']]
                    print(f"Fusion Output: {fusion_pred}    BERT Output:{bert_output}       True Label:{true_label}")

                bert_predictions.extend(bert_preds)
                bert_targets.extend(labels.cpu().numpy().tolist())

                current_fusion_acc = accuracy_score(fusion_targets, fusion_predictions)
                current_bert_acc = accuracy_score(bert_targets, bert_predictions) if bert_predictions else None
                val_loader.set_postfix({
                    'Fusion Acc': f'{current_fusion_acc:.4f}',
                    'BERT Acc': f'{current_bert_acc:.4f}' if current_bert_acc is not None else 'N/A'
                })

            except Exception as e:
                print(f"Error in validation: {str(e)}")
                continue

    if fusion_predictions:
        fusion_acc = accuracy_score(fusion_targets, fusion_predictions)
        fusion_f1_macro = f1_score(fusion_targets, fusion_predictions, average='macro')
        fusion_f1_weighted = f1_score(fusion_targets, fusion_predictions, average='weighted')
    else:
        fusion_acc = fusion_f1_macro = fusion_f1_weighted = 0

    if bert_predictions:
        bert_acc = accuracy_score(bert_targets, bert_predictions)
        bert_f1_macro = f1_score(bert_targets, bert_predictions, average='macro')
        bert_f1_weighted = f1_score(bert_targets, bert_predictions, average='weighted')
    else:
        bert_acc = bert_f1_macro = bert_f1_weighted = None

    return fusion_acc, fusion_f1_macro, fusion_f1_weighted, bert_acc, bert_f1_macro, bert_f1_weighted


def test_model(model, test_loaders, criterion, device, bert):
    bert = get_bert()
    model.eval()
    test_results = {}

    for name, test_loader_tuple in test_loaders.items():
        print(f"\nTesting {name} dataset...")

        if isinstance(test_loader_tuple, tuple):
            test_loader, specific_model = test_loader_tuple
            total_samples = len(test_loader.dataset)
            batch_size = test_loader.batch_size
            print(f"Total samples: {total_samples}")
            for batch in tqdm(test_loader, desc=f"Processing {name} samples",
                              total=total_samples, unit='samples',
                              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} samples [{elapsed}<{remaining}]'):
                pass
            results = validate(specific_model, test_loader, criterion, device)
        else:
            test_loader = test_loader_tuple
            total_samples = len(test_loader.dataset)
            batch_size = test_loader.batch_size
            print(f"Total samples: {total_samples}")
            for batch in tqdm(test_loader, desc=f"Processing {name} samples",
                              total=total_samples, unit='samples',
                              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} samples [{elapsed}<{remaining}]'):
                pass
            results = validate(model, test_loader, criterion, device)

        test_results[name] = {
            'Fusion Acc': results[0],
            'Fusion F1 Macro': results[1],
            'Fusion F1 Weighted': results[2],
            'BERT Acc': results[3],
            'BERT F1 Macro': results[4],
            'BERT F1 Weighted': results[5]
        }

        print(f"\n{name} Results:")
        print(f"Fusion Model - Accuracy: {results[0] * 100:.2f}%, "
              f"F1 Macro: {results[1] * 100:.2f}%, F1 Weighted: {results[2] * 100:.2f}%")
        if results[3] is not None:
            print(f"BERT Model - Accuracy: {results[3] * 100:.2f}%, "
                  f"F1 Macro: {results[4] * 100:.2f}%, F1 Weighted: {results[5] * 100:.2f}%")

    return test_results


def main():
    cfg = Config()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset_2015 = MultimodalDataset('data/IJCAI2019_data/twitter2015/train.tsv',
                                           'data/IJCAI2019_data/twitter2015_images',
                                           file_type='tsv', transform=transform)
    val_dataset_2015 = MultimodalDataset('data/IJCAI2019_data/twitter2015/dev.tsv',
                                         'data/IJCAI2019_data/twitter2015_images',
                                         file_type='tsv', transform=transform)
    test_dataset_2015 = MultimodalDataset('data/IJCAI2019_data/twitter2015/test.tsv',
                                          'data/IJCAI2019_data/twitter2015_images',
                                          file_type='tsv', transform=transform)

    train_dataset_2017 = MultimodalDataset('data/IJCAI2019_data/twitter2017/train.tsv',
                                           'data/IJCAI2019_data/twitter2017_images',
                                           file_type='tsv', transform=transform)
    val_dataset_2017 = MultimodalDataset('data/IJCAI2019_data/twitter2017/dev.tsv',
                                         'data/IJCAI2019_data/twitter2017_images',
                                         file_type='tsv', transform=transform)
    test_dataset_2017 = MultimodalDataset('data/IJCAI2019_data/twitter2017/test.tsv',
                                          'data/IJCAI2019_data/twitter2017_images',
                                          file_type='tsv', transform=transform)

    train_dataset_masad = MultimodalDataset('datasets/masad/train.tsv',
                                            'data/MASAD_imgs',
                                            file_type='tsv', transform=transform)
    val_dataset_masad = MultimodalDataset('datasets/masad/dev.tsv',
                                          'data/MASAD_imgs',
                                          file_type='tsv', transform=transform)
    test_dataset_masad = MultimodalDataset('datasets/masad/test.tsv',
                                           'data/MASAD_imgs',
                                           file_type='tsv', transform=transform)

    train_dataset_mvsa_s = MultimodalDataset('datasets/mvsa-s/train.tsv',
                                             'data/MVSA-S_data',
                                             file_type='tsv', transform=transform)
    val_dataset_mvsa_s = MultimodalDataset('datasets/mvsa-s/dev.tsv',
                                           'data/MVSA-S_data',
                                           file_type='tsv', transform=transform)
    test_dataset_mvsa_s = MultimodalDataset('datasets/mvsa-s/test.tsv',
                                            'data/MVSA-S_data',
                                            file_type='tsv', transform=transform)

    train_dataset_mvsa_m = MultimodalDataset('datasets/mvsa-m/train.tsv',
                                             'data/MVSA-M_data',
                                             file_type='tsv', transform=transform)
    val_dataset_mvsa_m = MultimodalDataset('datasets/mvsa-m/dev.tsv',
                                           'data/MVSA-M_data',
                                           file_type='tsv', transform=transform)
    test_dataset_mvsa_m = MultimodalDataset('datasets/mvsa-m/test.tsv',
                                            'data/MVSA-M_data',
                                            file_type='tsv', transform=transform)

    train_dataset_tumemo = MultimodalDataset('datasets/tumemo/train.tsv',
                                             'data/TumEmo_data',
                                             file_type='tsv', transform=transform)
    val_dataset_tumemo = MultimodalDataset('datasets/tumemo/dev.tsv',
                                           'data/TumEmo_data',
                                           file_type='tsv', transform=transform)
    test_dataset_tumemo = MultimodalDataset('datasets/tumemo/test.tsv',
                                            'data/TumEmo_data',
                                            file_type='tsv', transform=transform)
    # train_dataset = ConcatDataset([train_dataset_2015, train_dataset_2017,
    #                                train_dataset_mvsa_s, train_dataset_mvsa_m])

    # train_loader1 = DataLoader(train_dataset, batch_size=64,  shuffle=True)
    # train_loader2 = DataLoader(train_dataset_masad, batch_size=64, shuffle=True)
    train_loader3 = DataLoader(train_dataset_tumemo, batch_size=128, shuffle=True)

    # model = FusionModel(cfg, classnames=['negative', 'neutral', 'positive'], feature_dim=512)
    # model = model.float()
    # model_masad = FusionModel(cfg, classnames=['negative', 'positive'], feature_dim=512)
    # model_masad = model_masad.float()
    model_tumemo = FusionModel(cfg, classnames=['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad'],
                               feature_dim=512)
    model_tumemo = model_tumemo.float()

    # val_loaders1 = {
    #     't2015': DataLoader(val_dataset_2015, batch_size=64, shuffle=False)
    #     't2017': DataLoader(val_dataset_2017, batch_size=64, shuffle=False),
    #     'MVSA-S': DataLoader(val_dataset_mvsa_s, batch_size=64, shuffle=False),
    #     'MVSA-M': DataLoader(val_dataset_mvsa_m, batch_size=64, shuffle=False),
    # }
    #
    # val_loaders2 = {
    #     'MASAD': (DataLoader(val_dataset_masad, batch_size=64, shuffle=False), model_masad)
    # }

    val_loaders3 = {
        'TumEmo': (DataLoader(val_dataset_tumemo, batch_size=128, shuffle=False), model_tumemo)
    }

    # test_loaders1 = {
    #     't2015': DataLoader(test_dataset_2015, batch_size=64, shuffle=False),
    #     't2017': DataLoader(test_dataset_2017, batch_size=64, shuffle=False),
    #     'MVSA-S': DataLoader(test_dataset_mvsa_s, batch_size=64, shuffle=False),
    #     'MVSA-M': DataLoader(test_dataset_mvsa_m, batch_size=64, shuffle=False)
    # }
    #
    # test_loaders2 = {
    #     'MASAD': (DataLoader(test_dataset_masad, batch_size=64, shuffle=False), model_masad)
    # }

    test_loaders3 = {
        'TumEmo': (DataLoader(test_dataset_tumemo, batch_size=128, shuffle=False), model_tumemo)
    }

    criterion = PMRLoss(lambda_ce=1, lambda_proto=1e-3)
    optimizer = torch.optim.SGD(model_tumemo.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_fusion_model(model, train_loader1, val_loaders1, criterion, optimizer, scheduler, num_epochs=20, device=device)
    # train_fusion_model(model_masad, train_loader2, val_loaders2, criterion, optimizer, scheduler, num_epochs=20, device=device)
    train_fusion_model(model_tumemo, train_loader3, val_loaders3, criterion, optimizer, scheduler, num_epochs=10,
                       device=device)

    # test_results1 = test_model(model, test_loaders1, criterion, device, LlamaIntegration())
    # test_results2 = test_model(model_masad, test_loaders2, criterion, device, LlamaIntegration())
    test_results3 = test_model(model_tumemo, test_loaders3, criterion, device, BertIntegration())

    headers = ['Dataset', 'Fusion Acc', 'Fusion F1 Macro', 'Fusion F1 Weighted',
               'BERT Acc', 'BERT F1 Macro', 'BERT F1 Weighted']
    test_table = []
    for dataset, results in test_results3.items():
        test_table.append([dataset] + list(results.values()))

    print("\nTest Results:")
    print(tabulate(test_table, headers=headers, floatfmt='.4f'))

    print("Testing completed.")


if __name__ == "__main__":
    main()





