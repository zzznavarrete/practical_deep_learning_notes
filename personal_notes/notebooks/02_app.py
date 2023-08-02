

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()


learn = load_learner('02_model.pkl')


categories = ('Dog', 'Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['./images/dog.jpg', './images/cat.jpg', './images/dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)


