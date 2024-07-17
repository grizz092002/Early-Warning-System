import mysql.connector
import pandas as pd
import json
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
from dotenv import load_dotenv
from data_processing import process_data
import os

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_secret_key'  # Required for flashing messages

MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 4306))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DB = os.getenv('MYSQL_DB', 'dengue')

def connect_db():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']

        try:
            db = connect_db()
            cursor = db.cursor()

            query = 'INSERT INTO user_acc (fullname, email, password) VALUES (%s, %s, %s)'
            cursor.execute(query, (fullname, email, password))
            db.commit()

            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return redirect(url_for('registration', error='Database connection failed'))
        finally:
            if 'db' in locals() and db.is_connected():
                cursor.close()
                db.close()
    
    error = request.args.get('error')
    return render_template('registration.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        print(f"Attempting to log in with Username: {username} and Password: {password}")

        try:
            db = connect_db()
            cursor = db.cursor(dictionary=True)

            # Check in user_acc table
            query_user = 'SELECT * FROM approved_acc WHERE email = %s'
            cursor.execute(query_user, (username,))
            user = cursor.fetchone()

            print(f"User query result: {user}")

            if user and user['password'] == password:
                print("User login successful.")
                return redirect(url_for('user_page'))

            # Check in admin_acc table
            query_admin = 'SELECT * FROM admin_acc WHERE username = %s'
            cursor.execute(query_admin, (username,))
            admin = cursor.fetchone()

            print(f"Admin query result: {admin}")

            if admin and admin['password'] == password:
                print("Admin login successful.")
                return redirect(url_for('admin_page'))

            # If no matching user or admin found
            flash('Invalid username or password')
            return redirect(url_for('login'))

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            flash('Database connection failed')
            return redirect(url_for('login'))

        finally:
            if 'db' in locals() and db.is_connected():
                cursor.close()
                db.close()

    error = request.args.get('error')
    return render_template('login.html', error=error)

@app.route('/user')
def user_page():
    selected_municipality = request.args.get('municipality')

    municipalities = ['Famy', 'Kalayaan', 'Mabitac', 'Paete', 'Pakil', 'Pangil', 'Santa Maria', 'Siniloan']
    user_plots = {}
    user_yearly_plots = {}

    # Process data for each municipality
    for municipality in municipalities:
        (_, _, _, _, municipality_plots, municipality_yearly_plots, _) = process_data(municipality)
        user_plots[municipality] = municipality_plots.get(municipality, "")
        user_yearly_plots.update(municipality_yearly_plots)

    # Process data for the selected municipality
    (month_distribution_html, age_gender_html, cases_per_municipality_html,
     cases_per_municipality_data, _, monthly_admission_pivot,
     blood_type_html) = process_data(selected_municipality)

    return render_template('user.html',
                           month_distribution_html=month_distribution_html,
                           age_gender_html=age_gender_html,
                           cases_per_municipality_html=cases_per_municipality_html,
                           plots=user_plots,
                           selected_municipality=selected_municipality,
                           cases_per_municipality_data=cases_per_municipality_data,
                           monthly_admission_pivot=monthly_admission_pivot,
                           blood_type_html=blood_type_html)


@app.route('/admin')
def admin_page():
    selected_municipality = request.args.get('municipality')

    municipalities = ['Famy', 'Kalayaan', 'Mabitac', 'Paete', 'Pakil', 'Pangil', 'Santa Maria', 'Siniloan']
    admin_plots = {}
    admin_yearly_plots = {}

    # Process data for each municipality
    for municipality in municipalities:
        (_, _, _, _, municipality_plots, municipality_yearly_plots, _) = process_data(municipality)
        admin_plots[municipality] = municipality_plots.get(municipality, "")
        admin_yearly_plots.update(municipality_yearly_plots)

    # Process data for the selected municipality
    (month_distribution_html, age_gender_html, cases_per_municipality_html,
     cases_per_municipality_data, _, monthly_admission_pivot,
     blood_type_html) = process_data(selected_municipality)

    return render_template('admin.html',
                           month_distribution_html=month_distribution_html,
                           age_gender_html=age_gender_html,
                           cases_per_municipality_html=cases_per_municipality_html,
                           plots=admin_plots,
                           selected_municipality=selected_municipality,
                           cases_per_municipality_data=cases_per_municipality_data,
                           monthly_admission_pivot=monthly_admission_pivot,
                           blood_type_html=blood_type_html)

@app.route('/update_plot')
def update_plot():
    municipality = request.args.get('municipality')
    
    if municipality:
        _, _, _, _, cases_per_municipality_html, plots, _, _ = process_data(municipality)
        return jsonify({
            'casesPerMunicipality': cases_per_municipality_html,
            'barangayPlot': plots.get(municipality, "")
        })
    else:
        return jsonify({'error': 'No municipality selected'})
    
@app.route('/api/users')
def get_users():
    try:
        db = connect_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, fullname, email, password, status FROM user_acc")
        users = cursor.fetchall()
        return jsonify(users)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return jsonify({'error': 'Database connection failed'})
    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()
            
@app.route('/api/approve_user', methods=['POST'])
def approve_user():
    data = request.json
    user_id = data['userId']
    fullname = data['fullname']
    email = data['email']
    password = data['password']

    try:
        db = connect_db()
        cursor = db.cursor()

        # Insert into approved_acc
        cursor.execute("INSERT INTO approved_acc (fullname, email, password) VALUES (%s, %s, %s)",
                       (fullname, email, password))
        db.commit()

        # Update status in user_acc
        cursor.execute("UPDATE user_acc SET status = 'approved' WHERE id = %s", (user_id,))
        db.commit()

        return jsonify({'status': 'success'})

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return jsonify({'error': 'Database operation failed'})

    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()

@app.route('/api/reject_user', methods=['POST'])
def reject_user():
    data = request.json
    user_id = data['userId']

    try:
        db = connect_db()
        cursor = db.cursor()

        # Update status in user_acc
        cursor.execute("UPDATE user_acc SET status = 'rejected' WHERE id = %s", (user_id,))
        db.commit()

        return jsonify({'status': 'success'})

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return jsonify({'error': 'Database operation failed'})

    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()

@app.route('/approve_accounts')
def approve_accounts():
    return render_template('approve_accounts.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/repository')
def repository():
    return render_template('repository.html')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/forgotpass')
def forgotpass():
    return render_template('forgotpass.html')

@app.route('/resetpass')
def resetpass():
    return render_template('resetpass.html')

@app.route('/')
def home():
    error = request.args.get('error')
    return render_template('home.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)
