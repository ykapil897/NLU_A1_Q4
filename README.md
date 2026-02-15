<h1>Sports vs Politics Text Classification</h1>

<h2>Project Overview</h2>
<p>
This project implements a binary text classifier that classifies news articles as 
<b>Sport</b> or <b>Politics</b>.
</p>

<p>
All machine learning algorithms and feature engineering techniques are implemented 
<strong>from scratch</strong> using only:
</p>

<ul>
  <li>math</li>
  <li>random</li>
  <li>collections</li>
</ul>

<p>No external ML libraries such as scikit-learn, numpy, pandas, or nltk were used.</p>

<hr>

<h2>Dataset</h2>

<p>
Dataset used: <b>BBC News Text Dataset</b>
</p>

<p>
File format:
</p>

<pre>
category,text
</pre>

<p>
Only the following categories are used:
</p>

<ul>
  <li><b>sport</b> → Label 0</li>
  <li><b>politics</b> → Label 1</li>
</ul>

<p>
Other categories (business, tech, entertainment) are ignored.
</p>

<hr>

<h2>Feature Representations (Implemented from Scratch)</h2>

<h3>1. Bag of Words (BoW)</h3>
<ul>
  <li>Counts frequency of words in each document</li>
  <li>Vocabulary limited to top frequent words (default: 3000)</li>
</ul>

<h3>2. TF-IDF</h3>
<ul>
  <li>IDF = log(N / (1 + document frequency))</li>
  <li>TF-IDF = TF × IDF</li>
</ul>

<h3>3. n-grams</h3>
<ul>
  <li>Unigrams + Bigrams</li>
  <li>Bigrams generated as word1_word2</li>
</ul>

<hr>

<h2>Machine Learning Algorithms (From Scratch)</h2>

<h3>1. Multinomial Naive Bayes</h3>
<ul>
  <li>Uses Laplace smoothing</li>
  <li>Log-probability computation</li>
</ul>

<h3>2. Logistic Regression</h3>
<ul>
  <li>Gradient descent optimization</li>
  <li>Sigmoid activation</li>
</ul>

<h3>3. K-Nearest Neighbors (KNN)</h3>
<ul>
  <li>Euclidean distance</li>
  <li>Majority voting</li>
</ul>

<hr>

<h2>Experiments Conducted</h2>

<p>
Total experiments:
</p>

<pre>
3 Feature Representations × 3 Algorithms = 9 Experiments
</pre>

<table border="1" cellpadding="5">
<tr>
<th>Feature</th>
<th>Naive Bayes</th>
<th>Logistic Regression</th>
<th>KNN</th>
</tr>
<tr>
<td>Bag of Words</td>
<td>~100%</td>
<td>~99%</td>
<td>~92%</td>
</tr>
<tr>
<td>TF-IDF</td>
<td>~66%</td>
<td>~89%</td>
<td>~88%</td>
</tr>
<tr>
<td>NGRAM</td>
<td>~99%</td>
<td>~98%</td>
<td>~93%</td>
</tr>
</table>

<p>
(Results may vary slightly depending on dataset size and random split.)
</p>

<hr>

<h2>Project Structure</h2>

<pre>
sports-politics-classifier/
│
├── data/
│   └── bbc-text.csv
│
├── dataset.py
├── preprocess.py
├── features.py
├── naive_bayes.py
├── logistic_regression.py
├── knn.py
├── evaluation.py
└── main.py
</pre>

<hr>

<h2>How to Run</h2>

<ol>
  <li>Place <code>bbc-text.csv</code> inside <code>data/</code></li>
  <li>Open terminal in project directory</li>
  <li>Run:</li>
</ol>

<pre>
python main.py
</pre>

<p>
To change feature representation, modify the vectorization line inside <code>main.py</code>.
</p>

<hr>

<h2>Implementation Details</h2>

<ul>
  <li>Vocabulary limited to most frequent words</li>
  <li>80% training, 20% testing split</li>
  <li>Dataset shuffled before split</li>
  <li>No external machine learning libraries used</li>
</ul>

<hr>

<h2>Limitations</h2>

<ul>
  <li>Basic preprocessing (no stemming or lemmatization)</li>
  <li>No cross-validation</li>
  <li>No regularization in logistic regression</li>
  <li>KNN slow for large datasets</li>
</ul>

<hr>

<h2>Future Improvements</h2>

<ul>
  <li>Add cross-validation</li>
  <li>Implement regularization</li>
  <li>Improve token filtering</li>
  <li>Extend to multi-class classification</li>
</ul>

<hr>

<h2>Author</h2>
<p>
Implemented entirely from scratch for academic submission.
</p>
