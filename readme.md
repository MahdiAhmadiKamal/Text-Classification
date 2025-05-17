# Text Classification 

Natural language processing (NLP) combines computational linguistics, machine learning, and deep learning models to process human language. One of the most important applications of NLP is Text Classification.
<img src="pics\nlp.webp" width="500">

Text Classification is the task of assigning a sentence or document an appropriate category. The categories depend on the chosen dataset and can range from topics.
<img src="pics\text-classification.avif" width="500">

This project is capable of classifying input sentences into five categories. For this purpose, [GloVe](https://nlp.stanford.edu/projects/glove/), as an unsupervised learning algorithm, is used for obtaining vector representations for words. The categories that the network is trained on are represented by five emojis. The following table shows the categories for sentences and their related emojis and labels.

<table>
  <tr>
    <td>Category</td>
    <td>Emoji</td>
    <td>Label</td>
  </tr>
  <tr>
    <td>Affection and love</td>
    <td>❤️</td>
    <td>0</td>
  </tr>
   <td>Sports and exercise</td>
    <td>🏀</td>
    <td>1</td>
  </tr>
    <td>Gladness and encouragement</td>
    <td>😀</td>
    <td>2</td>
  </tr>
    <td>Despair and discouragement</td>
    <td>😔</td>
    <td>3</td>
  </tr>
      <td>Food and nutrition</td>
    <td>🍴</td>
    <td>4</td>
  </tr>
</table>

## How to install
Run this command:
```
pip install -r requirements.txt
```

## How to run
+ Download (glove.6B.zip)[https://nlp.stanford.edu/data/glove.6B.zip] as the pre-trained word vector
+ Unzip the downloaded file using the following command:
```
unzip -q /PATH/TO/glove.6B.zip -d glove.6B
```
+ Run the following command:
```
python Emoji_Text_Classification.py --sentence "Your sentence." --dimension 50
```
NOTE: As `dimension`, you can enter a preferred dimension for feature vectors: 50, 100, 200 or 300

## Results
### Without Dropout
<table>
  <tr>
    <td>Feature vector dimensions</td>
    <td>Train loss</td>
    <td>Train accuracy</td>
    <td>Test loss</td>
    <td>Test accuracy</td>
    <td>Inference Time</td>
  </tr>
  <tr>
    <td>50d</td>
    <td>0.71</td>
    <td>0.78</td>
    <td>0.72</td>
    <td>0.77</td>
    <td>0.109</td>
  </tr>
   <td>100d</td>
    <td>0.65</td>
    <td>0.80</td>
    <td>0.66</td>
    <td>0.80</td>
    <td>0.102</td>
  </tr>
    <td>200d</td>
    <td>0.47</td>
    <td>0.85</td>
    <td>0.49</td>
    <td>0.86</td>
    <td>0.106</td>
  </tr>
    <td>300d</td>
    <td>0.43</td>
    <td>0.90</td>
    <td>0.43</td>
    <td>0.89</td>
    <td>0.098</td>
  </tr>
</table>

### With Dropout=0.4
<table>
  <tr>
    <td>Feature vector dimensions</td>
    <td>Train loss</td>
    <td>Train accuracy</td>
    <td>Test loss</td>
    <td>Test accuracy</td>
    <td>Inference Time</td>
  </tr>
  <tr>
    <td>50d</td>
    <td>0.81</td>
    <td>0.75</td>
    <td>0.82</td>
    <td>0.73</td>
    <td>0.107</td>
  </tr>
   <td>100d</td>
    <td>0.74</td>
    <td>0.78</td>
    <td>0.75</td>
    <td>0.77</td>
    <td>0.111</td>
  </tr>
    <td>200d</td>
    <td>0.55</td>
    <td>0.85</td>
    <td>0.57</td>
    <td>0.84</td>
    <td>0.091</td>
  </tr>
    <td>300d</td>
    <td>0.47</td>
    <td>0.89</td>
    <td>0.48</td>
    <td>0.88</td>
    <td>0.112</td>
  </tr>
</table>