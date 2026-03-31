  """
  evaluate.py —— 在 Fashion-MNIST 测试集上评估训练好的 GDNClassifier。

  用法：
      python evaluate.py                          # 默认加载 gdn_fashion_mnist.pth
      python evaluate.py --ckpt path/to/model.pth # 指定权重路径
  """

  import os
  import sys
  import argparse

  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader
  from torchvision import datasets, transforms

  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from gated_deltanet import GDNClassifier

  CONFIG = {
      "img_size":      28,
      "patch_size":    4,
      "hidden_dim":    64,
      "num_heads":     4,
      "num_layers":    3,
      "mlp_expansion": 2,
      "num_classes":   10,
      "batch_size":    128,
  }

  FASHION_CLASSES = [
      "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
  ]


  def get_test_loader(batch_size: int = 128) -> DataLoader:
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.2860,), (0.3530,)),
      ])
      data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
      test_set = datasets.FashionMNIST(
          root=data_dir, train=False, download=True, transform=transform,
      )
      return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


  @torch.no_grad()
  def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
      model.eval()
      correct = total = 0
      class_correct = [0] * CONFIG["num_classes"]
      class_total   = [0] * CONFIG["num_classes"]

      for images, labels in loader:
          images, labels = images.to(device), labels.to(device)
          preds = model(images).argmax(dim=-1)
          correct += (preds == labels).sum().item()
          total   += labels.size(0)
          for c in range(CONFIG["num_classes"]):
              mask = labels == c
              class_correct[c] += (preds[mask] == labels[mask]).sum().item()
              class_total[c]   += mask.sum().item()

      overall_acc = correct / total
      per_class_acc = {
          FASHION_CLASSES[c]: class_correct[c] / class_total[c]
          for c in range(CONFIG["num_classes"])
      }
      return {"overall_acc": overall_acc, "per_class_acc": per_class_acc}


  def main():
      parser = argparse.ArgumentParser(description="Evaluate GDNClassifier on Fashion-MNIST")
      parser.add_argument(
          "--ckpt",
          default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_fashion_mnist.pth"),
          help="模型权重路径（默认：gdn_fashion_mnist.pth）",
      )
      args = parser.parse_args()

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(f"Device: {device}")

      model = GDNClassifier(
          img_size=CONFIG["img_size"],
          patch_size=CONFIG["patch_size"],
          hidden_dim=CONFIG["hidden_dim"],
          num_heads=CONFIG["num_heads"],
          num_layers=CONFIG["num_layers"],
          mlp_expansion=CONFIG["mlp_expansion"],
          num_classes=CONFIG["num_classes"],
      ).to(device)

      if not os.path.exists(args.ckpt):
          raise FileNotFoundError(f"找不到权重文件：{args.ckpt}，请先运行 train_fashion_mnist.py")

      model.load_state_dict(torch.load(args.ckpt, map_location=device))
      print(f"已加载权重：{args.ckpt}\n")

      test_loader = get_test_loader(CONFIG["batch_size"])
      results = evaluate(model, test_loader, device)

      print(f"{'='*45}")
      print(f"  整体测试准确率: {results['overall_acc']:.4f} ({results['overall_acc']*100:.2f}%)")
      print(f"{'='*45}")
      print("  各类别准确率：")
      for cls_name, acc in results["per_class_acc"].items():
          bar = "█" * int(acc * 20)
          print(f"  {cls_name:<15s} {acc:.4f}  {bar}")
      print(f"{'='*45}")


  if __name__ == "__main__":
      main()
