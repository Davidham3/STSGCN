"""Training entry point based on Lightning CLI.

Usage:
    python scripts/train.py fit --config config/PEMS04/individual_GLU_mask_emb.yaml
    python scripts/train.py fit --config ... --trainer.fast_dev_run true

After ``fit``, the best checkpoint (lowest val/loss) is automatically loaded
and evaluated on the test set with metrics in the original data scale.
"""

from pytorch_lightning.cli import LightningCLI

from stsgcn_torch import STSGCN, STSGCNDataModule


class STSGCNCLI(LightningCLI):
    def before_fit(self) -> None:
        # Replicate the original mxnet run: PolyScheduler needs the total
        # number of training steps, which depends on dataset size + epochs.
        self.datamodule.setup("fit")
        self.model.set_scheduler_params(
            num_train_samples=len(self.datamodule.train_x),
            batch_size=self.datamodule.batch_size,
            max_epochs=self.trainer.max_epochs,
        )


def main() -> None:
    cli = STSGCNCLI(
        STSGCN,
        STSGCNDataModule,
        save_config_kwargs={"overwrite": True},
    )
    # After fit, evaluate the best checkpoint (selected by ModelCheckpoint on val/loss).
    if cli.subcommand == "fit" and not cli.trainer.fast_dev_run:
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
