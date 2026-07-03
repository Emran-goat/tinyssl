# Optional import for PyTorch Hub: torch.hub.load("Emran-goat/tinyssl", "tinyssl_base")
dependencies = ["torch", "torchvision"]

from tinyssl.models.students import TinySSLBase, TinySSLTiny, TinySSLCNN


def tinyssl_base(pretrained=False, **kwargs):
    model = TinySSLBase(**kwargs)
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/tinyssl/tinyssl-base-flowers102/resolve/main/checkpoint_300.pt",
                map_location="cpu",
            )
        )
    return model


def tinyssl_tiny(pretrained=False, **kwargs):
    model = TinySSLTiny(**kwargs)
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/tinyssl/tinyssl-tiny-flowers102/resolve/main/checkpoint_300.pt",
                map_location="cpu",
            )
        )
    return model
