# Predictive-modeling-project
# 📦 Import libraries import numpy as np from sklearn.linear_model import LinearRegression  # 📊 Sample data (house size vs price) X = np.array([[500], [1000], [1500], [2000]])  # 🏠 size y = np.array([50, 100, 150, 200])  # 💰 price  # 🧠 Model training model = LinearRegression() model.fit(X, y)  # 🔮 Prediction print("💡 Prediction:..............


Here’s a **complete Predictive Modeling + Machine Learning roadmap with hands-on mini codes (Python)**, including **regression, classification, clustering, ensemble methods, tuning, and mini projects**. All code snippets include **# comments + emojis** for clarity.

---

# 🚀 1. Introduction to Machine Learning

Machine Learning (ML) is a field where models learn patterns from data to make predictions or decisions.

### 📌 Types of ML

* 🎯 Supervised Learning (labeled data)
* 🧠 Unsupervised Learning (unlabeled data)
* 🔁 Reinforcement Learning (reward-based learning)

---

# 📊 2. Supervised Learning

## 📈 Linear Regression (Simple + Multiple)

```python
# 📦 Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# 📊 Sample data (house size vs price)
X = np.array([[500], [1000], [1500], [2000]])  # 🏠 size
y = np.array([50, 100, 150, 200])  # 💰 price

# 🧠 Model training
model = LinearRegression()
model.fit(X, y)

# 🔮 Prediction
print("💡 Prediction:", model.predict([[1200]]))
```

---

## 📉 Ridge & Lasso Regression

```python
# 📦 Import models
from sklearn.linear_model import Ridge, Lasso

# 🧠 Ridge Regression (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# 🧠 Lasso Regression (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

print("🏁 Ridge Prediction:", ridge.predict([[1200]]))
print("🏁 Lasso Prediction:", lasso.predict([[1200]]))
```

---

# 🎯 3. Classification

## 📌 Logistic Regression

```python
# 📦 Import
from sklearn.linear_model import LogisticRegression

# 🧪 Sample data (study hours → pass/fail)
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print("🎓 Pass prediction:", model.predict([[3.5]]))
```

---

## 📍 KNN (K-Nearest Neighbors)

```python
# 📦 Import
from sklearn.neighbors import KNeighborsClassifier

# 🧠 Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print("📍 KNN Prediction:", knn.predict([[4]]))
```

---

## 🌳 Decision Tree

```python
# 🌳 Import
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X, y)

print("🌳 Decision Tree:", tree.predict([[2.5]]))
```

---

## 🧠 Support Vector Machine (SVM)

```python
# 📦 Import
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X, y)

print("🧠 SVM Prediction:", svm.predict([[3]]))
```

---

# 🌲 4. Ensemble Techniques

## 🌳 Random Forest

```python
# 🌲 Import
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)

print("🌲 Random Forest:", rf.predict([[3]]))
```

---

## ⚡ AdaBoost

```python
# ⚡ Import
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=10)
ada.fit(X, y)

print("⚡ AdaBoost:", ada.predict([[3]]))
```

---

# 🧪 5. Naive Bayes

```python
# 📦 Import
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X, y)

print("🧪 Naive Bayes:", nb.predict([[3]]))
```

---

# 🧩 6. Unsupervised Learning

## 📊 K-Means Clustering

```python
# 📦 Import
from sklearn.cluster import KMeans

data = [[1], [2], [3], [10], [11], [12]]

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

print("📊 Cluster Labels:", kmeans.labels_)
```

---

## 🌐 Hierarchical Clustering

```python
# 📦 Import
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=2)
labels = hc.fit_predict(data)

print("🌐 Hierarchical Clusters:", labels)
```

---

## 🔻 PCA (Dimensionality Reduction)

```python
# 📦 Import
from sklearn.decomposition import PCA
import numpy as np

X = np.random.rand(10, 5)  # 📊 5D data

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("🔻 Reduced Shape:", X_reduced.shape)
```

---

# 📉 7. Underfitting vs Overfitting

```python
# 🎯 Concept: model complexity tradeoff

# 🧠 Simple model → underfitting ❌
# 🧠 Complex model → overfitting ❌
# ⚖️ Balanced model → best ✔️
```

---

# 🔧 8. Hyperparameter Tuning

```python
# 📦 Import
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'max_depth': [2, 3, 5, 10]}  # 🔧 tuning

grid = GridSearchCV(DecisionTreeClassifier(), params, cv=3)
grid.fit(X, y)

print("🔧 Best Params:", grid.best_params_)
```

---

# ⚖️ 9. Bias vs Variance

```python
# 📉 High Bias → simple model ❌
# 📈 High Variance → overfitting ❌
# ⚖️ Sweet spot → balanced model ✔️
```

---

# 🧪 10. Mini Project Ideas

## 📧 Spam Classifier

```python
# 🧠 Idea: classify email as spam/not spam
# 📦 Use: Naive Bayes + TF-IDF
```

## 🏠 House Price Prediction

```python
# 📊 Use: Linear Regression / Random Forest
```

## ❤️ Heart Disease Prediction

```python
# 🧠 Use: Logistic Regression / SVM
```

---

# 🧠 11. Full Predictive Modeling Mini Project (Example)

## 🏆 Student Exam Prediction

```python
# 📦 Import
import numpy as np
from sklearn.linear_model import LogisticRegression

# 📊 Data: hours studied vs pass/fail
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])

# 🧠 Train model
model = LogisticRegression()
model.fit(X, y)

# 🔮 Predict
print("🎓 Will student pass?:", model.predict([[4.5]]))
```

---

Here’s a **complete Spam Detector using Machine Learning (Python)** 🚀
This is a real-world NLP classification project using **TF-IDF + Naive Bayes** (classic and very effective for spam detection).

---

# 📩 🧠 Spam Detector using Machine Learning

## 🧾 Step 1: Install Required Libraries

```python
# 📦 Install (run in terminal)
# pip install pandas scikit-learn
```

---

# 📊 Step 2: Import Libraries

```python id="spam001"
# 🧠 Spam Detection ML Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
```

---

# 📁 Step 3: Sample Dataset

(You can replace this with a real SMS dataset like SMS Spam Collection)

```python id="spam002"
# 📩 Simple dataset (spam = 1, ham = 0)

data = {
    "text": [
        "Win a free lottery now",
        "Congratulations you won a prize",
        "Hey, are we meeting today?",
        "Call me when you are free",
        "You have been selected for a reward",
        "Let’s go for dinner tonight"
    ],
    "label": [1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
print(df)
```

---

# ✂️ Step 4: Train-Test Split

```python id="spam003"
# 📊 Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)
```

---

# 🧠 Step 5: Build ML Pipeline (TF-IDF + Naive Bayes)

```python id="spam004"
# 🤖 Build spam detection model

model = Pipeline([
    ("tfidf", TfidfVectorizer()),      # 🔤 Convert text → numbers
    ("nb", MultinomialNB())            # 🧠 Classification model
])

# 🚀 Train model
model.fit(X_train, y_train)
```

---

# 🔮 Step 6: Predictions

```python id="spam005"
# 📩 Test messages
test_messages = [
    "Congratulations! You won a free iPhone",
    "Are we still meeting tomorrow?",
    "Claim your lottery prize now"
]

# 🔍 Predict
predictions = model.predict(test_messages)

for msg, pred in zip(test_messages, predictions):
    print("📩", msg)
    print("➡️ Spam" if pred == 1 else "➡️ Not Spam")
    print("-" * 40)
```

---

# 📊 Step 7: Model Accuracy

```python id="spam006"
# 📈 Accuracy check
y_pred = model.predict(X_test)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
```

---

# 🧠 How This Works

## ⚙️ Pipeline:

1. ✉️ Text input (email/SMS)
2. 🔤 TF-IDF converts text → numbers
3. 🧠 Naive Bayes learns patterns
4. 📊 Model classifies:

   * 1 → Spam 🚨
   * 0 → Not Spam ✅

---

# 📈 Why Naive Bayes?

✔ Fast
✔ Works great with text
✔ High accuracy for spam detection
✔ Industry standard baseline model

---

# 🚀 Real-World Upgrade Ideas

If you want to level this up:

## 🔥 Advanced Version

* 📊 Use real dataset (SMS Spam Collection)
* 🤖 Try Logistic Regression / SVM
* 🧠 Use Word Embeddings (Word2Vec / BERT)
* 🌐 Deploy using Flask / Streamlit

---

# 🌐 Mini Project Extension (Optional UI idea)

```python
# 💡 Streamlit app idea
# streamlit run app.py

# input box → predict spam/not spam
```

---

Here’s a **complete Email Spam Detector using a real dataset (SMS Spam Collection)** with proper ML workflow 🚀
We’ll use a real-world dataset, TF-IDF, and a strong classifier.

---

# 📩 🧠 EMAIL SPAM DETECTOR (REAL DATASET)

We’ll use the **SMS Spam Collection Dataset (UCI)**
👉 It contains real labeled messages: spam / ham (not spam)

---

# 📦 Step 1: Install Libraries

```python id="spam100"
# 📦 Install if needed
# pip install pandas scikit-learn numpy matplotlib seaborn
```

---

# 📥 Step 2: Load REAL Dataset

We will download it directly from UCI repository.

```python id="spam101"
import pandas as pd

# 📡 Load dataset from URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

print(df.head())
```

---

# 🧹 Step 3: Data Preprocessing

```python id="spam102"
# 🔄 Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("📊 Dataset shape:", df.shape)
print(df["label"].value_counts())
```

---

# ✂️ Step 4: Train-Test Split

```python id="spam103"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)
```

---

# 🧠 Step 5: Build ML Model (TF-IDF + Logistic Regression)

We use Logistic Regression (better than Naive Bayes for accuracy in many cases)

```python id="spam104"
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 🤖 Spam detection pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),  # 🔤 text → numbers
    ("clf", LogisticRegression(max_iter=1000))         # 🧠 classifier
])

# 🚀 Train model
model.fit(X_train, y_train)
```

---

# 🔮 Step 6: Make Predictions

```python id="spam105"
# 📩 Test emails
emails = [
    "Congratulations! You won a free iPhone. Click here now",
    "Hey, are we still meeting tomorrow?",
    "Urgent!!! Your bank account is locked. Verify now",
    "Let’s have lunch this weekend"
]

# 🔍 Predict
predictions = model.predict(emails)

for email, pred in zip(emails, predictions):
    print("\n📩 Email:", email)
    print("➡️ Result:", "🚨 SPAM" if pred == 1 else "✅ NOT SPAM")
```

---

# 📊 Step 7: Model Evaluation

```python id="spam106"
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Report:\n", classification_report(y_test, y_pred))
```

---

# 📈 Step 8: Confusion Matrix (Visualization)

```python id="spam107"
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Spam Detection Confusion Matrix")
plt.show()
```

---

# 🧠 How This System Works

## ⚙️ Pipeline:

1. 📩 Email text input
2. 🔤 TF-IDF converts text → numeric vectors
3. 🧠 Logistic Regression learns patterns
4. 🚨 Classifies:

   * 1 → Spam
   * 0 → Not Spam

---

# 📊 Why This Model Works Well

✔ Real dataset (industry style)
✔ Handles noisy text
✔ Logistic Regression is strong for text classification
✔ TF-IDF captures word importance

---

# 🚀 Upgrade Ideas (Advanced Level)

If you want to level up this project:

## 🔥 1. Deep Learning Spam Detector

* LSTM (RNN)
* BERT transformer model

## 🌐 2. Deploy as Web App

* Flask / FastAPI backend
* Streamlit UI

## 📱 3. Real Email Integration

* Gmail API
* Live spam filtering system

## 🧠 4. Advanced NLP

* Word2Vec / GloVe embeddings
* Attention-based models

---

Here’s a **complete Sentiment Analysis System using a real dataset + ML pipeline** 🚀
We’ll build a model that classifies text into:

* 😊 Positive
* 😐 Neutral (optional extension)
* 😡 Negative

We’ll use a **real Twitter sentiment dataset + TF-IDF + Logistic Regression** (strong baseline model).

---

# 🧠💬 SENTIMENT ANALYSIS SYSTEM (REAL ML PROJECT)

---

# 📦 Step 1: Install Libraries

```python id="sent100"
# 📦 Install if needed
# pip install pandas scikit-learn matplotlib seaborn
```

---

# 📥 Step 2: Load REAL Dataset

We’ll use a public Twitter sentiment dataset.

```python id="sent101"
import pandas as pd

# 📡 Real dataset (Sentiment140 sample)
url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"

df = pd.read_csv(url)

print(df.head())
```

---

# 🧹 Step 3: Keep Only Relevant Columns

```python id="sent102"
# 🧾 Keep only text + sentiment
df = df[["sentiment", "content"]]

print(df["sentiment"].value_counts())
```

---

# 🎯 Step 4: Convert Labels (Binary Sentiment)

We simplify into:

* Positive 😊
* Negative 😡

```python id="sent103"
# 🔄 Map emotions to sentiment
def convert_sentiment(label):
    if label in ["happiness", "love", "fun", "relief"]:
        return 1  # 😊 Positive
    else:
        return 0  # 😡 Negative

df["label"] = df["sentiment"].apply(convert_sentiment)

df = df[["content", "label"]]
print(df.head())
```

---

# ✂️ Step 5: Train-Test Split

```python id="sent104"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42
)
```

---

# 🧠 Step 6: Build ML Model (TF-IDF + Logistic Regression)

```python id="sent105"
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 🤖 Sentiment model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# 🚀 Train model
model.fit(X_train, y_train)
```

---

# 🔮 Step 7: Make Predictions

```python id="sent106"
# 💬 Test sentences
texts = [
    "I love this product, it's amazing!",
    "This is the worst experience ever",
    "I feel very happy and excited today",
    "I hate this so much"
]

predictions = model.predict(texts)

for text, pred in zip(texts, predictions):
    print("\n💬 Text:", text)
    print("➡️ Sentiment:", "😊 Positive" if pred == 1 else "😡 Negative")
```

---

# 📊 Step 8: Model Evaluation

```python id="sent107"
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Report:\n", classification_report(y_test, y_pred))
```

---

# 📈 Step 9: Confusion Matrix Visualization

```python id="sent108"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("📊 Sentiment Analysis Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---

# 🧠 How This System Works

## ⚙️ Pipeline:

1. 💬 Input text (tweet/review)
2. 🔤 TF-IDF converts text → numeric vectors
3. 🧠 Logistic Regression learns sentiment patterns
4. 😊/😡 classification output

---

# 📊 Why This Works Well

✔ Real dataset (Twitter emotions)
✔ TF-IDF captures word importance
✔ Logistic Regression is strong for NLP baseline
✔ Fast + scalable

---

# 🚀 Real-World Applications

This system is used in:

* 📱 Social media monitoring
* 🛒 Product review analysis
* 📊 Brand reputation tracking
* 🎬 Movie review classification
* 💬 Customer feedback systems

---

# 🔥 Upgrade Ideas (Advanced Level)

## 🤖 1. Deep Learning Version

* LSTM sentiment model
* BERT transformer model (state-of-the-art)

## 🌐 2. Web App Deployment

* Streamlit dashboard
* Flask API

## 📊 3. Advanced Features

* Neutral sentiment class
* Emoji sentiment detection
* Aspect-based sentiment analysis

---

# 📦 Import
from sklearn.linear_model import LinearRegression

# 🏠 Training data
X = [[1000],[1500],[2000],[2500]]
y = [50,75,100,125]

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[1800]])
print(prediction)

from sklearn.linear_model import LinearRegression

X = [
 [1000,2,1],
 [1500,3,2],
 [2000,4,2],
 [2500,5,3]
]

y=[50,75,100,125]

model=LinearRegression()
model.fit(X,y)

print(model.predict([[1800,3,2]]))

from sklearn.linear_model import Ridge

ridge=Ridge(alpha=1.0)
ridge.fit(X,y)

from sklearn.linear_model import Lasso

lasso=Lasso(alpha=0.1)
lasso.fit(X,y)

from sklearn.linear_model import LogisticRegression

X=[[1],[2],[3],[4],[5]]
y=[0,0,0,1,1]

model=LogisticRegression()
model.fit(X,y)

print(model.predict([[4]]))

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

print(knn.predict([[4]]))

from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier()
tree.fit(X,y)

print(tree.predict([[4]]))

from sklearn.svm import SVC

svm=SVC(kernel='linear')
svm.fit(X,y)

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
nb.fit(X,y)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(
    n_estimators=100
)

rf.fit(X,y)

from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(
    n_estimators=100
)

ada.fit(X,y)


from sklearn.cluster import KMeans

data=[
 [25,50000],
 [28,55000],
 [60,150000],
 [58,140000]
]

model=KMeans(
 n_clusters=2
)

model.fit(data)

print(model.labels_)

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(
 n_clusters=2
)

labels=hc.fit_predict(data)

print(labels)

from sklearn.decomposition import PCA
import numpy as np

X=np.random.rand(100,10)

pca=PCA(n_components=2)

reduced=pca.fit_transform(X)

print(reduced.shape)

from statsmodels.tsa.arima.model import ARIMA

model=ARIMA(data,order=(5,1,0))

model_fit=model.fit()

forecast=model_fit.forecast(30)

print(forecast)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

model=Sequential()

model.add(
 LSTM(50,input_shape=(10,1))
)

model.add(Dense(1))

model.compile(
 optimizer='adam',
 loss='mse'
)

from sklearn.model_selection import GridSearchCV

params={
 "max_depth":[3,5,10],
 "min_samples_split":[2,5,10]
}

grid=GridSearchCV(
 tree,
 params,
 cv=5
)

grid.fit(X,y)

print(grid.best_params_)

Annual Cost:
₹20–35 Crore/year





