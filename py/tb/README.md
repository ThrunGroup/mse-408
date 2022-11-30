# TODO

- Clarify engine runtime
  - Fundamentally, there needs to be a relationship between the environment and
    the representation required by the model
    - How can we abstract this with types without requiring information about
      NeuralNetworks?
  - There also needs to be a relationship between the loss and the training
    method employed by the model
  - Parameterization? Trainable?
  - Create a protocol for GFN models
  - When to run predict
  - When to run training_step() -> triggered by DataLoader?
- Use named tensors: https://pytorch.org/docs/stable/named_tensor.html
- Use nested tensors: `torch.nested.nested_tensor([t1, t2]).to_padded_tensor(-1)`
- Multiple protocols: https://stackoverflow.com/questions/60732924/how-to-specify-that-a-python-object-must-be-two-types-at-once
- Inference: https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_basic.html
