from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from flask import flash, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Load data
trending_products = pd.read_csv('models/trending_products.csv')
train_data = pd.read_csv('models/clean_data.csv')

# database configuration
app.secret_key = 'password'
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/retail"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Model for Signup
class Signup(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    username = db.Column(db.String(100),nullable = False)
    email = db.Column(db.String(100),nullable = False)
    password = db.Column(db.String(100), nullable=False)

# Model for Signin
class Signin(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    username = db.Column(db.String(100),nullable = False)
    password = db.Column(db.String(100), nullable=False)


def content_based_generator(train_data, item_name, top_n=25):
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the dataset.")
        return pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand'])

    tfid_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfid_vectorizer.fit_transform(train_data['Tags'])

    cosine_similarity_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarity_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n + 1]
    recommended_items_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_items_indices][['Name', 'ReviewCount', 'Brand']]
    return recommended_items_details

# Sample image URLs and prices
random_image_urls = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png"
]

prices = [20, 50, 150, 70, 100, 200, 106, 100, 75, 120]


# Helper function to truncate product names or descriptions
def truncate(text, length=50):
    return text if len(text) <= length else text[:length] + "..."


# Routes
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/main")
def main():
    return render_template('main.html')


@app.route("/trending")
def trending():
    trending_data = trending_products.head(8).to_dict(orient='records')
    image_urls = [random.choice(random_image_urls) for _ in range(len(trending_data))]
    prices = [random.randint(50, 300) for _ in range(len(trending_data))]

    return render_template('trending.html',
                           trending_products=trending_data,
                           image_urls=image_urls,
                           prices=prices)


@app.route("/index")
def indexredirect():
    return render_template('index.html')

# Routes
@app.route("/signup", methods=['POST', 'GET'])
def signup_route():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()
        flash("Signup successful! You can now sign in.", "success")
        return redirect(url_for('index'))

    # If it's a GET request, redirect to homepage or show form
    return redirect(url_for('index'))

# Routes
@app.route("/signin", methods=['POST', 'GET'])
def signin_route():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        new_signin = Signin(username=username, password=password)
        db.session.add(new_signin)
        db.session.commit()
        flash("Signin successful", "success")
        return redirect(url_for('index'))

    # If it's a GET request, redirect to homepage or show form
    return redirect(url_for('index'))

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendation():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr', 8))

        content_based_rec = content_based_generator(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            flash("No recommendations found. Please try another product.", "danger")
            return redirect(url_for("main"))

        image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
        prices_list = [random.choice(prices) for _ in range(len(content_based_rec))]

        # ✅ Zip it here
        product_data = list(zip(content_based_rec.iterrows(), image_urls, prices_list))

        return render_template("main.html",
                               product_data=product_data,
                               truncate=truncate,
                               content_based_rec=content_based_rec,  # optional: in case you still check it
                               selected_count=nbr)
    return redirect(url_for("main"))



if __name__ == '__main__':
    app.run(debug=True)
