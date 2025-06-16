import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from .dataset import MultimodalDataset
from .model import FusionModel
from .loss import PMRLoss
from .config import Config
from tqdm import tqdm
from sklearn.metrics import f1_score
from .llama_integration import LlamaIntegration
import warnings

warnings.filterwarnings("ignore",
                        message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_fusion_model(model, train_loader, val_loaders, criterion, optimizer, scheduler, num_epochs, device):
    llama = LlamaIntegration()
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for batch in train_bar:
            images, texts, labels = batch['image'].to(device), batch['text'], batch['label'].to(device)

            optimizer.zero_grad()

            try:
                with torch.cuda.amp.autocast():
                    logits, prototypes, features, similarities, text_features, explanations = model(images, texts)
                    # print(f"Batch - Logits: shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")

                    loss, ce_loss, proto_loss = criterion(logits, prototypes, features, labels)
                    # print(f"Batch - Loss: {loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, Proto Loss: {proto_loss.item():.4f}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                train_bar.set_postfix(
                    {'Loss': f'{train_loss / train_total:.4f}', 'Acc': f'{100. * train_correct / train_total:.2f}%'})
            except RuntimeError as e:
                print(f"Error in batch: {str(e)}")
                continue

        scheduler.step()

        model.eval()
        val_accs = []
        val_f1s = []
        llama_accs = []
        llama_f1s = []
        for name, val_loader in val_loaders.items():
            if name in ['MASAD', 'TumEmo']:
                val_loader, specific_model = val_loader
                fusion_acc, fusion_f1, llama_acc, llama_f1 = validate(specific_model, val_loader, criterion, device,
                                                                      llama, dataset_name=name)
            else:
                fusion_acc, fusion_f1, llama_acc, llama_f1 = validate(model, val_loader, criterion, device, llama,
                                                                      dataset_name=name)
            val_accs.append(f'{name} Fusion Acc: {fusion_acc:.2f}%')
            val_f1s.append(f'{name} Fusion F1: {fusion_f1:.4f}')
            llama_accs.append(f'{name} LLaMA Acc: {llama_acc:.2f}%')
            llama_f1s.append(f'{name} LLaMA F1: {llama_f1:.4f}')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / train_total:.4f}, '
              f'Train Acc: {100. * train_correct / train_total:.2f}%, '
              f'{", ".join(val_accs)}, {", ".join(val_f1s)}, '
              f'{", ".join(llama_accs)}, {", ".join(llama_f1s)}')


def validate(model, val_loader, criterion, device, llama, dataset_name=''):
    model.eval()
    val_correct = 0
    llama_correct = 0
    val_total = 0
    all_fusion_preds = []
    all_llama_preds = []
    all_labels = []

    val_bar = tqdm(val_loader, desc='Validating')

    with torch.no_grad():
        for batch in val_bar:
            images, texts, labels = batch['image'].to(device), batch['text'], batch['label'].to(device)

            logits, prototypes, features, similarities, text_features, explanations = model(images, texts)
            _, predicted = logits.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            all_fusion_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(texts)):
                confidence = logits[i].softmax(0)[predicted[i]].item()
                prototype_similarity = similarities[i][predicted[i]].item()

                if dataset_name == 'MASAD':
                    label_map = ['negative', 'positive']
                elif dataset_name == 'TumEmo':
                    label_map = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
                else:
                    label_map = ['negative', 'neutral', 'positive']

                fusion_output = (
                    f"Predicted label: {label_map[predicted[i].item()]}, "
                    f"Confidence: {confidence:.4f}, "
                    f"Prototype similarity: {prototype_similarity:.4f}, "
                    f"Text: {texts[i][:100]}..., "
                    f"Explanation: {explanations[i]}"
                )

                llama_result = llama.process_fusion_output(fusion_output)
                if llama_result and 'sentiment' in llama_result:
                    llama_sentiment = llama_result['sentiment'].lower()
                    if dataset_name == 'MASAD':
                        llama_prediction = 1 if llama_sentiment == 'positive' else 0
                    elif dataset_name == 'TumEmo':
                        llama_prediction = label_map.index(llama_sentiment.capitalize())
                    else:
                        llama_prediction = {'positive': 2, 'neutral': 1, 'negative': 0}.get(llama_sentiment,
                                                                                            predicted[i].item())
                else:
                    llama_prediction = predicted[i].item()
                all_llama_preds.append(llama_prediction)
                if llama_prediction == labels[i].item():
                    llama_correct += 1

            val_bar.set_postfix({'Fusion Acc': f'{100. * val_correct / val_total:.2f}%'})

    fusion_acc = 100. * val_correct / val_total
    fusion_f1 = f1_score(all_labels, all_fusion_preds, average='macro')
    llama_acc = 100. * llama_correct / val_total
    llama_f1 = f1_score(all_labels, all_llama_preds, average='macro')

    return fusion_acc, fusion_f1, llama_acc, llama_f1


def test_model(model, test_loaders, criterion, device, llama):
    model.eval()
    test_results = {}
    for name, test_loader in test_loaders.items():
        if name == 'MASAD':
            test_loader, masad_model = test_loader
            fusion_acc, fusion_f1, llama_acc, llama_f1 = validate(masad_model, test_loader, criterion, device, llama)
        else:
            fusion_acc, fusion_f1, llama_acc, llama_f1 = validate(model, test_loader, criterion, device, llama)
        test_results[name] = {
            'Fusion Acc': fusion_acc,
            'Fusion F1': fusion_f1,
            'LLaMA Acc': llama_acc,
            'LLaMA F1': llama_f1
        }
    return test_results


def main():
    cfg = Config()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # train_dataset_2015 = MultimodalDataset('data/IJCAI2019_data/twitter2015/train.tsv',
    #                                        'data/IJCAI2019_data/twitter2015_images',
    #                                        file_type='tsv', transform=transform)
    # val_dataset_2015 = MultimodalDataset('data/IJCAI2019_data/twitter2015/dev.tsv',
    #                                      'data/IJCAI2019_data/twitter2015_images',
    #                                      file_type='tsv', transform=transform)
    # test_dataset_2015 = MultimodalDataset('data/IJCAI2019_data/twitter2015/test.tsv',
    #                                       'data/IJCAI2019_data/twitter2015_images',
    #                                       file_type='tsv', transform=transform)
    #
    # train_dataset_2017 = MultimodalDataset('data/IJCAI2019_data/twitter2017/train.tsv',
    #                                        'data/IJCAI2019_data/twitter2017_images',
    #                                        file_type='tsv', transform=transform)
    # val_dataset_2017 = MultimodalDataset('data/IJCAI2019_data/twitter2017/dev.tsv',
    #                                      'data/IJCAI2019_data/twitter2017_images',
    #                                      file_type='tsv', transform=transform)
    # test_dataset_2017 = MultimodalDataset('data/IJCAI2019_data/twitter2017/test.tsv',
    #                                       'data/IJCAI2019_data/twitter2017_images',
    #                                       file_type='tsv', transform=transform)
    #
    # train_dataset_masad = MultimodalDataset('datasets/masad/train.tsv',
    #                                         'data/MASAD_imgs',
    #                                         file_type='tsv', transform=transform)
    # val_dataset_masad = MultimodalDataset('datasets/masad/dev.tsv',
    #                                       'data/MASAD_imgs',
    #                                       file_type='tsv', transform=transform)
    # test_dataset_masad = MultimodalDataset('datasets/masad/test.tsv',
    #                                        'data/MASAD_imgs',
    #                                        file_type='tsv', transform=transform)
    #
    # train_dataset_mvsa_s = MultimodalDataset('datasets/mvsa-s/train.tsv',
    #                                          'data/MVSA-S_data',
    #                                          file_type='tsv', transform=transform)
    # val_dataset_mvsa_s = MultimodalDataset('datasets/mvsa-s/dev.tsv',
    #                                        'data/MVSA-S_data',
    #                                        file_type='tsv', transform=transform)
    # test_dataset_mvsa_s = MultimodalDataset('datasets/mvsa-s/test.tsv',
    #                                         'data/MVSA-S_data',
    #                                         file_type='tsv', transform=transform)
    #
    # train_dataset_mvsa_m = MultimodalDataset('datasets/mvsa-m/train.tsv',
    #                                          'data/MVSA-M_data',
    #                                          file_type='tsv', transform=transform)
    # val_dataset_mvsa_m = MultimodalDataset('datasets/mvsa-m/dev.tsv',
    #                                        'data/MVSA-M_data',
    #                                        file_type='tsv', transform=transform)
    # test_dataset_mvsa_m = MultimodalDataset('datasets/mvsa-m/test.tsv',
    #                                         'data/MVSA-M_data',
    #                                         file_type='tsv', transform=transform)

    train_dataset_tumemo = MultimodalDataset('datasets/tumemo/train.tsv',
                                             'data/TumEmo_data',
                                             file_type='tsv', transform=transform)
    val_dataset_tumemo = MultimodalDataset('datasets/tumemo/dev.tsv',
                                           'data/TumEmo_data',
                                           file_type='tsv', transform=transform)
    test_dataset_tumemo = MultimodalDataset('datasets/tumemo/test.tsv',
                                            'data/TumEmo_data',
                                            file_type='tsv', transform=transform)

    train_dataset = ConcatDataset([train_dataset_2015, train_dataset_2017,
                                   train_dataset_mvsa_s, train_dataset_mvsa_m])

    train_loader1 = DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_loader2 = DataLoader(train_dataset_masad, batch_size=64, shuffle=True)
    train_loader3 = DataLoader(train_dataset_tumemo, batch_size=64, shuffle=True)

    model = FusionModel(cfg, classnames=['negative', 'neutral', 'positive'], feature_dim=512)
    model = model.float()
    model_masad = FusionModel(cfg, classnames=['negative', 'positive'], feature_dim=512)
    model_masad = model_masad.float()
    model_tumemo = FusionModel(cfg, classnames=['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad'],
                               feature_dim=512)
    model_tumemo = model_tumemo.float()

    val_loaders1 = {
        't2015': DataLoader(val_dataset_2015, batch_size=64, shuffle=False)
        't2017': DataLoader(val_dataset_2017, batch_size=64, shuffle=False),
        'MVSA-S': DataLoader(val_dataset_mvsa_s, batch_size=64, shuffle=False),
        'MVSA-M': DataLoader(val_dataset_mvsa_m, batch_size=64, shuffle=False),
    }

    val_loaders2 = {
        'MASAD': (DataLoader(val_dataset_masad, batch_size=64, shuffle=False), model_masad)
    }

    val_loaders3 = {
        'TumEmo': (DataLoader(val_dataset_tumemo, batch_size=64, shuffle=False), model_tumemo)
    }

    test_loaders1 = {
        't2015': DataLoader(test_dataset_2015, batch_size=64, shuffle=False),
        't2017': DataLoader(test_dataset_2017, batch_size=64, shuffle=False),
        'MVSA-S': DataLoader(test_dataset_mvsa_s, batch_size=64, shuffle=False),
        'MVSA-M': DataLoader(test_dataset_mvsa_m, batch_size=64, shuffle=False)
    }

    test_loaders2 = {
        'MASAD': (DataLoader(test_dataset_masad, batch_size=64, shuffle=False), model_masad)
    }

    test_loaders3 = {
        'TumEmo': (DataLoader(test_dataset_tumemo, batch_size=64, shuffle=False), model_tumemo)
    }

    criterion = PMRLoss(lambda_ce=1, lambda_proto=1e-3)
    optimizer = torch.optim.SGD(model_tumemo.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_fusion_model(model, train_loader1, val_loaders1, criterion, optimizer, scheduler, num_epochs=20,
                       device=device)
    train_fusion_model(model_masad, train_loader2, val_loaders2, criterion, optimizer, scheduler, num_epochs=20,
                       device=device)
    train_fusion_model(model_tumemo, train_loader3, val_loaders3, criterion, optimizer, scheduler, num_epochs=20,
                       device=device)

    test_results1 = test_model(model, test_loaders1, criterion, device, LlamaIntegration())
    test_results2 = test_model(model_masad, test_loaders2, criterion, device, LlamaIntegration())
    test_results3 = test_model(model_tumemo, test_loaders3, criterion, device, LlamaIntegration())

    for dataset, results in test_results1.items():
        print(f"\nTest Results for {dataset}:")
        print(f"Fusion Accuracy: {results['Fusion Acc']:.2f}%")
        print(f"Fusion F1 Score: {results['Fusion F1']:.4f}")
        print(f"LLaMA Accuracy: {results['LLaMA Acc']:.2f}%")
        print(f"LLaMA F1 Score: {results['LLaMA F1']:.4f}")

    for dataset, results in test_results2.items():
        print(f"\nTest Results for {dataset}:")
        print(f"Fusion Accuracy: {results['Fusion Acc']:.2f}%")
        print(f"Fusion F1 Score: {results['Fusion F1']:.4f}")
        print(f"LLaMA Accuracy: {results['LLaMA Acc']:.2f}%")
        print(f"LLaMA F1 Score: {results['LLaMA F1']:.4f}")

    for dataset, results in test_results3.items():
        print(f"\nTest Results for {dataset}:")
        print(f"Fusion Accuracy: {results['Fusion Acc']:.2f}%")
        print(f"Fusion F1 Score: {results['Fusion F1']:.4f}")
        print(f"LLaMA Accuracy: {results['LLaMA Acc']:.2f}%")
        print(f"LLaMA F1 Score: {results['LLaMA F1']:.4f}")

    print("Testing completed.")


if __name__ == "__main__":
    main()
