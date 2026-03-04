import os
import re
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_logs(filename):
    plt.clf()
    with open(filename, 'r') as f:
        content = f.read()
    epochs, accs, losses, asrs, asr_losses = [], [], [], [], []
    train_accs, train_losses = [], []
    regex = (
        r"Epoch (?P<epoch>\d+)\s.*?"
        r"Train Acc: (?P<train_acc>[\d\.]+)\s.*?"
        r"Train loss: (?P<train_loss>[\d\.]+)\s.*?"
        r"(?:Test|Val) Acc: (?P<test_acc>[\d\.]+)\s.*?"
        r"(?:Test|Val) loss: (?P<test_loss>[\d\.]+)"
        r"(?:\s.*?ASR: (?P<asr>[\d\.]+))?"
        r"(?:\s.*?ASR loss: (?P<asr_loss>[\d\.]+))?"
    )

    for match in re.finditer(regex, content):
        epochs.append(int(match.group('epoch')))
        train_accs.append(float(match.group('train_acc')))
        train_losses.append(float(match.group('train_loss')))
        accs.append(float(match.group('test_acc')))
        losses.append(float(match.group('test_loss')))

        asr = match.group('asr')
        asr_loss = match.group('asr_loss')
        asrs.append(float(asr) if asr else None)
        asr_losses.append(float(asr_loss) if asr_loss else None)

    return epochs, train_accs, train_losses, accs, losses, asrs, asr_losses


def plot_accuracy(filename):
    epochs, _, _, accs, _, asr, _ = parse_logs(filename)

    plt.plot(epochs, accs, label='Accuracy')

    if any(asr):
        plt.plot(epochs, asr, label='ASR', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename[:-4] + ".png")


def plot_convergence(filename):
    """Plot a 2x2 grid: Train Loss, Eval Loss, Eval Acc, ASR."""
    epochs, train_accs, train_losses, test_accs, test_losses, asrs, asr_losses = parse_logs(filename)
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(os.path.basename(filename).replace('.txt', ''), fontsize=13)

    axes[0, 0].plot(epochs, train_losses, color='tab:blue')
    axes[0, 0].set_title('Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes[0, 1].plot(epochs, test_losses, color='tab:orange')
    axes[0, 1].set_title('Eval Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    axes[1, 0].plot(epochs, test_accs, color='tab:green')
    axes[1, 0].set_title('Eval Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    if any(a is not None for a in asrs):
        asr_vals = [a if a is not None else 0.0 for a in asrs]
        axes[1, 1].plot(epochs, asr_vals, color='tab:red')
    axes[1, 1].set_title('Attack Success Rate (ASR)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ASR')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = filename[:-4] + "_convergence.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_distribution(train_data, client_idcs, n_clients, dataset,
                            distribution, output_dir=None, num_adv=0,
                            dirichlet_alpha=None, filename_prefix=None):
    titleid_dict = {
        "iid": "Balanced_IID",
        "class-imbalanced_iid": "Class-imbalanced_IID",
        "non-iid": "Quantity-imbalanced_Dirichlet_Non-IID",
        "pat": "Balanced_Pathological_Non-IID",
        "imbalanced_pat": "Quantity-imbalanced_Pathological_Non-IID",
    }
    ds_name = "CIFAR-10" if dataset == "CIFAR10" else dataset
    title_id = ds_name + " " + titleid_dict[distribution]

    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    labels = train_data.targets
    n_classes = int(labels.max() + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    ax.hist(label_distribution, stacked=True,
            bins=np.arange(-0.5, n_clients + 1.5, 1),
            label=[f"Class {c}" for c in range(n_classes)],
            rwidth=0.5, zorder=10)

    tick_labels = []
    for c_id in range(n_clients):
        lbl = str(c_id)
        if c_id < num_adv:
            lbl += "*"
        tick_labels.append(lbl)
    ax.set_xticks(np.arange(n_clients))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Client ID (* = adversary)", fontsize=20)
    ax.set_ylabel("Number of Training Samples", fontsize=20)

    alpha_str = f" (alpha={dirichlet_alpha})" if dirichlet_alpha is not None else ""
    ax.set_title(f"{title_id}{alpha_str}\nLabel Distribution Across Clients",
                 fontsize=18)
    rotation_degree = 45 if n_clients > 30 else 0
    ax.tick_params(axis='x', rotation=rotation_degree, labelsize=16)
    ax.legend(loc="best", prop={'size': 11}).set_zorder(100)
    ax.grid(linestyle='--', axis='y', zorder=0)
    fig.tight_layout()

    if output_dir is None:
        output_dir = "./logs"
    os.makedirs(output_dir, exist_ok=True)
    alpha_tag = f"_alpha{dirichlet_alpha}" if dirichlet_alpha is not None else ""
    file_stem = f"noniid{alpha_tag}_distribution"
    if filename_prefix:
        file_stem = f"{filename_prefix}_{file_stem}"
    save_path = os.path.join(output_dir, f"{file_stem}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _unnormalize(img_tensor, mean, std):
    """Reverse torchvision Normalize for display."""
    img = img_tensor.clone()
    for c, m, s in zip(range(img.shape[0]), mean, std):
        img[c] = img[c] * s + m
    return img.clamp(0, 1)


def visualize_dba_triggers(attacker, args, output_dir=None, filename_prefix=None):
    """
    Render a 1x6 grid: Original | Local 0-3 | Global trigger.
    Works on the first sample of the adversary's local dataset.
    """
    dataset = attacker.train_dataset
    sample_img, sample_label = dataset[0]
    if not isinstance(sample_img, torch.Tensor):
        from torchvision import transforms
        trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std),
        ])
        sample_img = trans(sample_img)
    if not isinstance(sample_label, (int, float)):
        sample_label = int(sample_label)

    original_label = sample_label
    target_label = attacker.target_label
    trigger = attacker.trigger
    trigger_nums = attacker.trigger_nums

    panels = []
    titles = []

    panels.append(_unnormalize(sample_img.clone(), args.mean, args.std))
    titles.append(f"Original\nClass {original_label}")

    for t_idx in range(trigger_nums):
        img_copy = sample_img.clone()
        attacker.setup_trigger_position(t_idx)
        pos = attacker.trigger_position
        t = trigger[t_idx]
        t_h, t_w = t.shape[-2], t.shape[-1]
        end_r = None if pos[0] + t_h == 0 else pos[0] + t_h
        end_c = None if pos[1] + t_w == 0 else pos[1] + t_w
        img_copy[:, pos[0]:end_r, pos[1]:end_c] = t
        panels.append(_unnormalize(img_copy, args.mean, args.std))
        titles.append(f"Local trigger {t_idx}\n{original_label} -> {target_label}")

    global_img = sample_img.clone()
    for t_idx in range(trigger_nums):
        attacker.setup_trigger_position(t_idx)
        pos = attacker.trigger_position
        t = trigger[t_idx]
        t_h, t_w = t.shape[-2], t.shape[-1]
        end_r = None if pos[0] + t_h == 0 else pos[0] + t_h
        end_c = None if pos[1] + t_w == 0 else pos[1] + t_w
        global_img[:, pos[0]:end_r, pos[1]:end_c] = t
    panels.append(_unnormalize(global_img, args.mean, args.std))
    titles.append(f"Global trigger\n{original_label} -> {target_label}")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    for ax, img, title in zip(axes, panels, titles):
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0).numpy(), cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.suptitle("DBA Trigger Visualization", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_dir is None:
        output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    file_name = "dba_triggers.png"
    if filename_prefix:
        file_name = f"{filename_prefix}_dba_triggers.png"
    save_path = os.path.join(output_dir, file_name)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plot_accuracy(
        "./logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_DBA_DeepSight_500_50_0.01_FedOpt.txt")
