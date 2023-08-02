### Notes on class 3 of [Practical deep learning] course

- For choose a particular neural net as architecture for my learner, I could list them with (if for instance I'm interested on convnext architectures):

```python 
timm.list_models('convnext*')

```
then I could place a particular neural net in my learner:

```
learn = vision_learner(dls, 'convnext_tiny_in22k', metrics=error_rate).to_fp16()
learn.fine_tune(3)
```

