from django.core.management.base import BaseCommand
from mlmini.classification.train import train_classification_model

class Command(BaseCommand):
    help = "分類モデルの学習（out/weightN を自動採番）"

    def add_arguments(self, parser):
        parser.add_argument("--data", dest="dataset_directory", required=True)
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--out", dest="output_directory", default="./out")
        parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])

    def handle(self, *args, **opts):
        output = train_classification_model(
            dataset_directory=opts["dataset_directory"],
            output_base_directory=opts["output_directory"],
            epochs=opts["epochs"],
            device=opts["device"],
        )
        self.stdout.write(self.style.SUCCESS(f"Saved to: {output}"))
