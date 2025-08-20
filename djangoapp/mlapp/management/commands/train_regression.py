from django.core.management.base import BaseCommand
from mlmini.regression.train import train_regression_model

class Command(BaseCommand):
    help = "回帰モデルの学習（out/weightN を自動採番）"

    def add_arguments(self, parser):
        parser.add_argument("--out", dest="output_directory", default="./out")
        parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])

    def handle(self, *args, **opts):
        output = train_regression_model(output_base_directory=opts["output_directory"], device=opts["device"])
        self.stdout.write(self.style.SUCCESS(f"Saved to: {output}"))
