# nlp-assignment-1-solved
**TO GET THIS SOLUTION VISIT:** [NLP Assignment 1 Solved](https://www.ankitcodinghub.com/product/nlp-assignment-aspect-term-polarity-classification-in-sentiment-analysis-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;123949&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;NLP Assignment 1 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
1 Introduction

The goal of this assignment is to implement a classifier that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences. The classifier takes as input 3 elements: a sentence, an aspect term occurring in the sentence, and its aspect category. For each input triple, it produces a polarity label: positive, negative or neutral.

2 Dataset

The dataset is in TSV format, one instance per line. As an example, here are 2 instances:

negative SERVICE#GENERAL Wait staff 0:10 Wait staff is blantently unappreciative of your business but its the best pie on the UWS!

positive FOOD#QUALITY pie 74:77 Wait staff is blantently unappreciative of your business but its the best pie on the UWS!

Each line contains 5 tab-separated fields: the polarity of the opinion (the ground truth polarity label), the aspect category on which the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the sentence in which the term occurs and the opinion is expressed.

For instance, in the first line, the opinion polarity regarding the target term “wait staff”, which has the aspect category SERVICE#GENERAL, is negative.In the example of the second line, the sentence is the same but the opinion is about a different aspect and a different target term (pie), and is positive.

There are 12 different aspects categories:

AMBIENCE#GENERAL

DRINKS#PRICES

DRINKS#QUALITY

DRINKS#STYLE_OPTIONS

FOOD#PRICES

FOOD#QUALITY

FOOD#STYLE_OPTIONS

LOCATION#GENERAL

RESTAURANT#GENERAL

RESTAURANT#MISCELLANEOUS

RESTAURANT#PRICES

SERVICE#GENERAL

The training set (filename: traindata.csv) has this format (5 fields) and contains 1503 lines, i.e. 1503 opinions. The classifier should be learned only from this training set.

A development dataset (filename: devdata.csv) is distributed to help you set up your classifier and estimate its performance. It has the same format as the training dataset. It has 376 lines, i.e. 376 opinions.

We will perform the final evaluation by measuring the accuracy of the classifier on a test dataset that is not distributed. The majority class of the dev set is about 70% (positive labels), and will be considered as a (weak) baseline.

3 How to proceed

1. Create a python environment and install/use python &gt;= 3.9.x (required). Besides the standard python modules, you can use the following libraries:

a. pytorch = 1.13.1

b. pytorch-lightning = 1.8.1

c. transformers = 4.22.2

d. datasets = 2.9.0 (just the library ‘datasets’, no labelled data)

e. sentencepiece = 0.1.97

f. scikit-learn = 1.2.0

g. numpy = 1.23.5

h. pandas = 1.5.3

i. nltk = 3.8.1

j. stanza = 1.4.2

2. Download the nlp_assignment.zip file and uncompress it to a dedicated root folder. The root folder will contain 2 subfolders:

a. data: contains traindata.csv and devdata.csv

b. src: contains 2 python files: tester.py, classifier.py

3. Implement your classifier by completing the “Classifier” class template in src/classifier.py, containing the following 2 methods:

a. The train method takes training data file and a dev data file as input, and trains the model on the specified device

b. The predict method takes a data file (e.g. devdata.csv), it should run on the specified device return a python list of predicted labels. The returned list contains the predicted labels in the same order as the corresponding examples in the input file

4. You can create new python files in the src subfolder, if needed to implement the classifier.

5. Run the model using the device specified as a parameter in the train() and predict() methods. Please do not use a default device (like ‘cuda’ or ‘cuda:0’)! Also, the model should not require more than 14GB of memory to run on the data (that’s the limit of the GPU device on which the program will be evaluated).

6. To check and test your classifier, cd to the src subfolder and run tester.py. It should run without errors, training the model on traindata.csv and evaluating it on devdata.csv, and reporting the accuracy measure.

7. Please do not modify tester.py! Your program must run successfully without having to modify this file.

8. The exact content of the deliverable is described in section 4 of this document

9. Your project deliverable must be a unique zip file (a compressed folder). No gz, or other compression format.

10. The name of the zip file must consist of the family names of the authors of the deliverable. Example: Clouseau_Holmes_Velasquez.zip

11. The zip file size should not exceed 3 MBs.

4 Deliverable Content

When uncompressed, the main folder must contain the following elements:

Element Description

README.txt

A plain text file that should contain:

1. Names of the students who contributed to the deliverable (max=4)

2. A clear and detailed description of the implemented classifier (type of classification model, input and feature representation, resources etc.)

3. The accuracy that you get on the dev dataset.

src

A subfolder containing ALL the python source files required to train and run your classifier using the unmodified tester.py, including the completed classifier.py file. You can put in this folder other code files (that you import from classifier.py), and any potential resources your model requires (e.g. list of polarity words or other manual features, if you use such features).

Please do not include pre-trained LM files in the deliverable (the deliverable size is limited to 3 MB).

Note:

– Please make sure that when you cd to the src subfolder and launch tester.py (unmodified!) with the python interpreter, it runs without errors: it trains the classifier on the train set and evaluates it on the dev dataset, outputting the average accuracy.

– You can use any type of models. However, note that classification methods based on pretrained, transformer-based language models usually yield better accuracy.

– You can also use deep models combined with explicit or implicit linguistic features, e.g.

polarity lexicons.
