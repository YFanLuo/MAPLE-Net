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
import warnings
from tabulate import tabulate
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore",
                        message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn.functional as F
from .feature_bank import FeatureBank

llama_integration = None
label_map = {
    0: "Angry",
    1: "Bored",
    2: "Calm",
    3: "Fear",
    4: "Happy",
    5: "Love",
    6: "Sad"
}


def get_llama():
    global llama_integration
    if llama_integration is None:
        print("Initializing LLaMA model...")
        llama_integration = LlamaIntegration()
    return llama_integration


def plot_test_accuracy(history, save_path='./plots/tumemo_few_221.png'):
    test_records = [record for record in history if record[2] == "Test"]
    epochs = sorted(list(set([int(record[0]) for record in test_records])))

    fusion_acc = [float(record[4]) for record in test_records]
    llama_acc = [float(record[5]) for record in test_records]
    fusion_f1_macro = [float(record[6]) for record in test_records]
    llama_f1_macro = [float(record[7]) for record in test_records]
    fusion_f1_weighted = [float(record[8]) for record in test_records]
    llama_f1_weighted = [float(record[9]) for record in test_records]

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, fusion_acc, 'b-', label='Fusion Accuracy')
    plt.plot(epochs, llama_acc, 'r--', label='LLaMA Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, fusion_f1_macro, 'b-', label='Fusion F1 Macro')
    plt.plot(epochs, llama_f1_macro, 'r--', label='LLaMA F1 Macro')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Macro')
    plt.title('Test F1 Macro over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, fusion_f1_weighted, 'b-', label='Fusion F1 Weighted')
    plt.plot(epochs, llama_f1_weighted, 'r--', label='LLaMA F1 Weighted')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Weighted')
    plt.title('Test F1 Weighted over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)
    print(f"\nTest accuracy plot saved to {save_path}")

    plt.close()


def train_fusion_model(model, train_loader, val_loaders, test_loaders, criterion, optimizer, scheduler, num_epochs=10,
                       device='cuda'):
    best_val_acc = 0
    best_epoch = 0
    history = []

    print("Initializing feature bank...")
    model.feature_bank = FeatureBank(device)
    model.feature_bank.build_bank(model, train_loader)

    headers = ['Epoch', 'Train Loss', 'Dataset Type', 'Dataset',
               'Fusion Acc', 'LLaMA Acc',
               'Fusion F1 Macro', 'LLaMA F1 Macro',
               'Fusion F1 Weighted', 'LLaMA F1 Weighted']

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
                fusion_acc, fusion_f1_macro, fusion_f1_weighted, \
                llama_acc, llama_f1_macro, llama_f1_weighted = validate(
                    specific_model, val_loader, criterion, device, epoch + 1
                )
            else:
                val_loader = val_loader_tuple
                fusion_acc, fusion_f1_macro, fusion_f1_weighted, \
                llama_acc, llama_f1_macro, llama_f1_weighted = validate(
                    model, val_loader, criterion, device, epoch + 1
                )

            val_results = [
                epoch + 1,
                f"{avg_train_loss:.4f}",
                "Validation",
                name,
                f"{fusion_acc:.4f}",
                f"{llama_acc:.4f}" if llama_acc is not None else "N/A",
                f"{fusion_f1_macro:.4f}",
                f"{llama_f1_macro:.4f}" if llama_f1_macro is not None else "N/A",
                f"{fusion_f1_weighted:.4f}",
                f"{llama_f1_weighted:.4f}" if llama_f1_weighted is not None else "N/A"
            ]
            results_table.append(val_results)
            history.append(val_results)

        # 测试阶段
        test_results = test_model(model, test_loaders, criterion, device, get_llama())
        for dataset, results in test_results.items():
            test_results = [
                epoch + 1,
                f"{avg_train_loss:.4f}",
                "Test",
                dataset,
                f"{results['Fusion Acc']:.4f}",
                f"{results['LLaMA Acc']:.4f}",
                f"{results['Fusion F1 Macro']:.4f}",
                f"{results['LLaMA F1 Macro']:.4f}",
                f"{results['Fusion F1 Weighted']:.4f}",
                f"{results['LLaMA F1 Weighted']:.4f}"
            ]
            results_table.append(test_results)
            history.append(test_results)

        print(f"\nEpoch {epoch + 1} Results:")
        print(tabulate(results_table, headers=headers, tablefmt="grid"))

        if scheduler is not None:
            scheduler.step()

        # 使用验证集结果更新最佳模型
        current_val_acc = fusion_acc
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch + 1
            print(f'\nNew best model saved! (Epoch {best_epoch}, Accuracy: {best_val_acc:.4f})')
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}')

    plot_test_accuracy(history)

    print("\nComplete Training History:")
    print(tabulate(history, headers=headers, tablefmt="grid",
                   floatfmt=".4f", numalign="right"))

    best_val_results = [r for r in history if r[0] == best_epoch and r[2] == "Validation"][0]
    best_test_results = [r for r in history if r[0] == best_epoch and r[2] == "Test"][0]

    print("\nBest Model Performance (Epoch {}):".format(best_epoch))
    print("\nValidation Results:")
    print(f"Fusion Accuracy: {best_val_results[4]}")
    print(f"Fusion F1 Macro: {best_val_results[6]}")
    print(f"Fusion F1 Weighted: {best_val_results[8]}")
    if best_val_results[5] != "N/A":
        print(f"LLaMA Accuracy: {best_val_results[5]}")
        print(f"LLaMA F1 Macro: {best_val_results[7]}")
        print(f"LLaMA F1 Weighted: {best_val_results[9]}")

    print("\nTest Results:")
    print(f"Fusion Accuracy: {best_test_results[4]}")
    print(f"Fusion F1 Macro: {best_test_results[6]}")
    print(f"Fusion F1 Weighted: {best_test_results[8]}")
    print(f"LLaMA Accuracy: {best_test_results[5]}")
    print(f"LLaMA F1 Macro: {best_test_results[7]}")
    print(f"LLaMA F1 Weighted: {best_test_results[9]}")

    return best_val_acc, best_epoch, history


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
    llama_predictions = []
    llama_targets = []
    llama = get_llama()

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
                llama_preds = []
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
                        print(f"Reference_samples: {reference_samples}       Reference_similarities:{ref_similarities}")
                        result = llama.process_fusion_output(fusion_output, reference_samples, ref_similarities)
                    else:
                        result = llama.process_fusion_output(fusion_output)

                    llama_preds.append(result['sentiment'])
                    print(f"Fusion Output: {fusion_pred}    LLaMA Output:{label_map[result['sentiment']]}       True Label:{true_label}")

                llama_predictions.extend(llama_preds)
                llama_targets.extend(labels.cpu().numpy().tolist())


            except Exception as e:
                print(f"Error in validation: {str(e)}")
                continue

    if fusion_predictions:
        fusion_acc = accuracy_score(fusion_targets, fusion_predictions)
        fusion_f1_macro = f1_score(fusion_targets, fusion_predictions, average='macro')
        fusion_f1_weighted = f1_score(fusion_targets, fusion_predictions, average='weighted')
    else:
        fusion_acc = fusion_f1_macro = fusion_f1_weighted = 0

    if llama_predictions:
        llama_acc = accuracy_score(llama_targets, llama_predictions)
        llama_f1_macro = f1_score(llama_targets, llama_predictions, average='macro')
        llama_f1_weighted = f1_score(llama_targets, llama_predictions, average='weighted')
    else:
        llama_acc = llama_f1_macro = llama_f1_weighted = None

    return fusion_acc, fusion_f1_macro, fusion_f1_weighted, llama_acc, llama_f1_macro, llama_f1_weighted


def test_model(model, test_loaders, criterion, device, llama):
    llama = get_llama()
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
            'LLaMA Acc': results[3],
            'LLaMA F1 Macro': results[4],
            'LLaMA F1 Weighted': results[5]
        }

        print(f"\n{name} Results:")
        print(f"Fusion Model - Accuracy: {results[0] * 100:.2f}%, "
              f"F1 Macro: {results[1] * 100:.2f}%, F1 Weighted: {results[2] * 100:.2f}%")
        if results[3] is not None:
            print(f"LLaMA Model - Accuracy: {results[3] * 100:.2f}%, "
                  f"F1 Macro: {results[4] * 100:.2f}%, F1 Weighted: {results[5] * 100:.2f}%")

    return test_results


def main():
    cfg = Config()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset_2015 = MultimodalDataset('compare_methods/data/IJCAI2019_data/twitter2015/train.tsv',
                                           'compare_methods/data/IJCAI2019_data/twitter2015_images',
                                           file_type='tsv', transform=transform)
    val_dataset_2015 = MultimodalDataset('compare_methods/data/IJCAI2019_data/twitter2015/dev.tsv',
                                         'compare_methods/data/IJCAI2019_data/twitter2015_images',
                                         file_type='tsv', transform=transform)
    test_dataset_2015 = MultimodalDataset('compare_methods/data/IJCAI2019_data/twitter2015/test.tsv',
                                          'compare_methods/data/IJCAI2019_data/twitter2015_images',
                                          file_type='tsv', transform=transform)

    # train_dataset_2017 = MultimodalDataset('compare_methods/data/IJCAI2019_data/twitter2017/train.tsv',
    #                                        'compare_methods/data/IJCAI2019_data/twitter2017_images',
    #                                        file_type='tsv', transform=transform)
    # val_dataset_2017 = MultimodalDataset('compare_methods/data/IJCAI2019_data/twitter2017/dev.tsv',
    #                                      'compare_methods/data/IJCAI2019_data/twitter2017_images',
    #                                      file_type='tsv', transform=transform)
    # test_dataset_2017 = MultimodalDataset('compare_methods/data/IJCAI2019_data/twitter2017/test.tsv',
    #                                       'compare_methods/data/IJCAI2019_data/twitter2017_images',
    #                                       file_type='tsv', transform=transform)
    #
    # train_dataset_masad = MultimodalDataset('compare_methods/datasets/masad/train.tsv',
    #                                         'compare_methods/data/MASAD_imgs',
    #                                         file_type='tsv', transform=transform)
    # val_dataset_masad = MultimodalDataset('compare_methods/datasets/masad/dev.tsv',
    #                                       'compare_methods/data/MASAD_imgs',
    #                                       file_type='tsv', transform=transform)
    # test_dataset_masad = MultimodalDataset('compare_methods/datasets/masad/test.tsv',
    #                                        'compare_methods/data/MASAD_imgs',
    #                                        file_type='tsv', transform=transform)
    #
    # train_dataset_mvsa_s = MultimodalDataset('compare_methods/datasets/mvsa-s/train.tsv',
    #                                          'compare_methods/data/MVSA-S_data',
    #                                          file_type='tsv', transform=transform)
    # val_dataset_mvsa_s = MultimodalDataset('compare_methods/datasets/mvsa-s/dev.tsv',
    #                                        'compare_methods/data/MVSA-S_data',
    #                                        file_type='tsv', transform=transform)
    # test_dataset_mvsa_s = MultimodalDataset('compare_methods/datasets/mvsa-s/test.tsv',
    #                                         'compare_methods/data/MVSA-S_data',
    #                                         file_type='tsv', transform=transform)
    #
    # train_dataset_mvsa_m = MultimodalDataset('compare_methods/datasets/mvsa-m/train.tsv',
    #                                          'compare_methods/data/MVSA-M_data',
    #                                          file_type='tsv', transform=transform)
    # val_dataset_mvsa_m = MultimodalDataset('compare_methods/datasets/mvsa-m/dev.tsv',
    #                                        'compare_methods/data/MVSA-M_data',
    #                                        file_type='tsv', transform=transform)
    # test_dataset_mvsa_m = MultimodalDataset('compare_methods/datasets/mvsa-m/test.tsv',
    #                                         'compare_methods/data/MVSA-M_data',
    #                                         file_type='tsv', transform=transform)
    #
    # train_dataset_tumemo = MultimodalDataset('compare_methods/datasets/tumemo/train_few2.tsv',
    #                                          'compare_methods/data/TumEmo_data',
    #                                          file_type='tsv', transform=transform)
    # val_dataset_tumemo = MultimodalDataset('compare_methods/datasets/tumemo/dev_few2.tsv',
    #                                        'compare_methods/data/TumEmo_data',
    #                                        file_type='tsv', transform=transform)
    # test_dataset_tumemo = MultimodalDataset('compare_methods/datasets/tumemo/dev_few1.tsv',
    #                                         'compare_methods/data/TumEmo_data',
    #                                         file_type='tsv', transform=transform)
    # train_dataset = ConcatDataset([train_dataset_2015, train_dataset_2017,
    #                                train_dataset_mvsa_s, train_dataset_mvsa_m])

    train_loader1 = DataLoader(train_dataset_2015, batch_size=64,  shuffle=True)
    # train_loader2 = DataLoader(train_dataset_masad, batch_size=64, shuffle=True)
    # train_loader3 = DataLoader(train_dataset_tumemo, batch_size=128, shuffle=True)

    model = FusionModel(cfg, classnames=['negative', 'neutral', 'positive'], feature_dim=512)
    model = model.float()
    # model_masad = FusionModel(cfg, classnames=['negative', 'positive'], feature_dim=512)
    # model_masad = model_masad.float()
    # model_tumemo = FusionModel(cfg, classnames=['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad'],
    #                            feature_dim=512)
    # model_tumemo = model_tumemo.float()

    val_loaders1 = {
        't2015': DataLoader(val_dataset_2015, batch_size=64, shuffle=False)
        # 't2017': DataLoader(val_dataset_2017, batch_size=64, shuffle=False),
        # 'MVSA-S': DataLoader(val_dataset_mvsa_s, batch_size=64, shuffle=False),
        # 'MVSA-M': DataLoader(val_dataset_mvsa_m, batch_size=64, shuffle=False),
    }
    #
    # val_loaders2 = {
    #     'MASAD': (DataLoader(val_dataset_masad, batch_size=64, shuffle=False), model_masad)
    # }

    # val_loaders3 = {
    #     'TumEmo': (DataLoader(val_dataset_tumemo, batch_size=128, shuffle=False), model_tumemo)
    # }

    test_loaders1 = {
        't2015': DataLoader(test_dataset_2015, batch_size=64, shuffle=False),
    #     't2017': DataLoader(test_dataset_2017, batch_size=64, shuffle=False),
    #     'MVSA-S': DataLoader(test_dataset_mvsa_s, batch_size=64, shuffle=False),
    #     'MVSA-M': DataLoader(test_dataset_mvsa_m, batch_size=64, shuffle=False)
    }
    #
    # test_loaders2 = {
    #     'MASAD': (DataLoader(test_dataset_masad, batch_size=64, shuffle=False), model_masad)
    # }

    # test_loaders3 = {
    #     'TumEmo': (DataLoader(test_dataset_tumemo, batch_size=128, shuffle=False), model_tumemo)
    # }

    criterion = PMRLoss(lambda_ce=1, lambda_proto=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_fusion_model(model, train_loader1, val_loaders1,test_loaders1, criterion, optimizer, scheduler, num_epochs=20, device=device)
    # train_fusion_model(model_masad, train_loader2, val_loaders2,test_loaders2, criterion, optimizer, scheduler, num_epochs=20, device=device)
    # train_fusion_model(model_tumemo, train_loader3, val_loaders3, test_loaders3, criterion, optimizer, scheduler,
    #                    num_epochs=40,
    #                    device=device)

    test_results1 = test_model(model, test_loaders1, criterion, device, LlamaIntegration())
    # test_results2 = test_model(model_masad, test_loaders2, criterion, device, LlamaIntegration())
    # test_results3 = test_model(model_tumemo, test_loaders3, criterion, device, LlamaIntegration())

    headers = ['Dataset', 'Fusion Acc', 'Fusion F1 Macro', 'Fusion F1 Weighted',
               'LLaMA Acc', 'LLaMA F1 Macro', 'LLaMA F1 Weighted']
    test_table = []
    for dataset, results in test_results3.items():
        test_table.append([dataset] + list(results.values()))

    print("\nTest Results:")
    print(tabulate(test_table, headers=headers, floatfmt='.4f'))

    print("Testing completed.")


if __name__ == "__main__":
    main()
