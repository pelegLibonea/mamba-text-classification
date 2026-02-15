import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf,load_state_dict_hf
from collections import namedtuple
import torch.nn as nn
import torch

from cfg.config import MambaConfig
from mamba.head import MambaClassificationHead

class MambaTextClassification(MambaLMHeadModel):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg = None,
        device = None,
        dtype = None,
        num_classes = 2,
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)
        
        # Create a classification head using MambaClassificationHead with input size of d_model and configurable number of classes.
        self.num_classes = num_classes
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=num_classes)
        
        del self.lm_head
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        # Pass input_ids through the backbone model to receive hidden_states.
        hidden_states = self.backbone(input_ids)
        
        # Take the mean of hidden_states along the second dimension to create a representative [CLS] feature.
        mean_hidden_states = hidden_states.mean(dim = 1)
        
        # Pass mean_hidden_states through the classification head to get logits.
        logits = self.classification_head(mean_hidden_states)
        
        if labels is None:
            ClassificationOuptput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOuptput(logits = logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            
            # Use CrossEntropyLoss loss function to compute the loss.
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            return ClassificationOuptput(loss = loss, logits = logits)
    def predict(self, text, tokenizer, id2label = None):
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device = "cuda")[None]
        with torch.no_grad():
            logits = self.forward(input_ids).logits[0]
            label = np.argmax(logits.cpu().numpy())
            
        if id2label is not None:
            return id2label[label]
        else:
            return label
    
    def save_pretrained(self, save_directory, base_model_name="state-spaces/mamba-130m", **kwargs):
        """
        Save the model and its configuration to a directory.
        Similar to the reference code's save mechanism.
        
        Args:
            save_directory: Directory to save the model
            base_model_name: Name of the base pretrained model (for later loading)
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_dict = {
            "num_classes": self.num_classes,
            "d_model": self.backbone.embedding.weight.shape[1],
            "vocab_size": self.backbone.embedding.weight.shape[0],
            "base_model_name": base_model_name,
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def load_pretrained_local(cls, save_directory, device = None, dtype = None):
        """
        Load a model from a local directory that was saved with save_pretrained.
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        num_classes = saved_config.get("num_classes", 2)
        base_model_name = saved_config.get("base_model_name", "state-spaces/mamba-130m")
        
        # Load the base pretrained model with correct num_classes
        model = cls.from_pretrained(
            base_model_name,
            device=device,
            dtype=dtype,
            num_classes=num_classes
        )
        
        # Load the saved state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {save_directory}")
        return model
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device = None, dtype = None, num_classes = 2, **kwargs):
        # Load the configuration from the pre-trained model.
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        
        # Initialize the model from the configuration and move it to the desired device and data type.
        model = cls(config, device = device, dtype = dtype, num_classes = num_classes, **kwargs)
        
        # Load the state of the pre-trained model.
        model_state_dict = load_state_dict_hf(pretrained_model_name, device = device, dtype = dtype)
        model.load_state_dict(model_state_dict , strict=False)
        
        # Print the newly initialized embedding parameters.
        print (" Newly initialized embedding :", 
              set(model.state_dict().keys()) - set(model_state_dict.keys())
        )

        return model.to(device)