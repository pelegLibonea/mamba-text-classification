# Future Enhancements

This document tracks potential improvements for future iterations. The current implementation is production-ready, but these enhancements could further improve code quality and maintainability.

## Potential Improvements

### 1. More Robust Class Detection (trainer.py, example_multiclass.py)
**Current:** Uses `hasattr()` to check for `num_classes` attribute
**Enhancement:** Use `getattr()` with default value for more robust error handling
```python
# Current
if hasattr(loaded_dataset["train"].features["label"], 'num_classes'):
    num_classes = loaded_dataset["train"].features["label"].num_classes

# Enhanced
num_classes = getattr(
    loaded_dataset["train"].features.get("label"), 
    'num_classes', 
    None
)
if num_classes is None:
    num_classes = len(set(loaded_dataset["train"]["label"]))
```

### 2. Decouple Save Logic from Internal Structure (mamba/model.py)
**Current:** Accesses `self.backbone.embedding.weight.shape` directly
**Enhancement:** Store d_model and vocab_size as instance attributes during initialization
```python
# In __init__
self.d_model = config.d_model
self.vocab_size = config.vocab_size

# In save_pretrained
config_dict = {
    "num_classes": self.num_classes,
    "d_model": self.d_model,
    "vocab_size": self.vocab_size,
    "base_model_name": base_model_name,
}
```

### 3. Dynamic Class Label Extraction (example_multiclass.py)
**Current:** Hardcoded class labels in print statement
**Enhancement:** Extract dynamically from dataset features
```python
# Current
print(f"Classes: World (0), Sports (1), Business (2), Sci/Tech (3)")

# Enhanced
try:
    class_names = loaded_dataset["train"].features["label"].names
    class_list = ", ".join([f"{name} ({i})" for i, name in enumerate(class_names)])
    print(f"Classes: {class_list}")
except (AttributeError, KeyError):
    print(f"Classes: {num_classes} total classes")
```

### 4. Configuration Management
**Enhancement:** Create a dedicated Config class for model configuration
- Centralizes all configuration parameters
- Makes it easier to serialize/deserialize
- Improves maintainability

### 5. Validation Utilities
**Enhancement:** Add runtime validation for model configuration
- Validate num_classes > 0
- Validate base_model_name format
- Check dataset compatibility before training

### 6. Extended Metrics
**Enhancement:** Add per-class metrics reporting
- Per-class precision, recall, F1
- Confusion matrix generation
- Class-specific performance analysis

### 7. Training Callbacks
**Enhancement:** Add custom callbacks for monitoring
- Class-balanced loss tracking
- Per-class accuracy during training
- Early stopping based on specific classes

## Implementation Priority

**High Priority:**
- None - current implementation is production-ready

**Medium Priority:**
- #2: Decouple save logic (improves maintainability)
- #1: More robust class detection (better error handling)

**Low Priority:**
- #3: Dynamic label extraction (nice-to-have)
- #4-7: Future features (not critical for core functionality)

## Notes

All these enhancements are **optional**. The current implementation:
- ✅ Is fully functional
- ✅ Follows best practices
- ✅ Is production-ready
- ✅ Has comprehensive documentation
- ✅ Maintains backward compatibility

These enhancements would add marginal improvements in specific edge cases or future maintenance scenarios.
