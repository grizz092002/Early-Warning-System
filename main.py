import mysql.connector
import pandas as pd
import json
import joblib
import os
import warnings
import sklearn
import pickle
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
from dotenv import load_dotenv
from datetime import datetime, timedelta
from data_processing import process_data
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError


app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_secret_key'  # Required for flashing messages

MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 4306))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DB = os.getenv('MYSQL_DB', 'dengue')

load_dotenv()

UPLOAD_FOLDER = 'C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def read_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding='utf-8')
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")


def connect_db():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    
model_path = 'C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/best_svm_model.pkl'

# Load the pre-trained model
model_path = 'C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/best_svm_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Process uploaded files
        files = ['file1', 'file2', 'file3', 'file4']
        dfs = {}
        
        for file_key in files:
            if file_key in request.files:
                file = request.files[file_key]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    dfs[file_key] = read_file(file_path)  # Use the updated read_file function

        # Check if all required files are uploaded
        if len(dfs) != len(files):
            flash('Please upload all required files.', 'error')
            return redirect(url_for('predict'))

        # Data processing and merging
        temperature_df = dfs['file1']
        dengue_df = dfs['file2']
        humidity_df = dfs['file3']
        mosquito_df = dfs['file4']
        
        # Process data
        dengue_df['DateOfEntry'] = pd.to_datetime(dengue_df['DateOfEntry'], format='%d/%m/%Y')
        temperature_df['Date'] = pd.to_datetime(temperature_df['Date'], format='%d/%m/%Y')
        mosquito_df['Trap Date'] = pd.to_datetime(mosquito_df['Trap Date'], format='%m/%d/%Y')
        humidity_df['Date'] = pd.to_datetime(humidity_df['Date'], format='%d/%m/%Y')

        dengue_df['DateMonthYear'] = dengue_df['DateOfEntry'].dt.to_period('M').dt.to_timestamp()

        # Aggregate daily and monthly data
        dengue_daily = dengue_df.groupby('DateMonthYear').size().reset_index(name='Cases')
        dengue_agg = dengue_df.groupby('DateMonthYear').agg({
            'Muncity': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Sex': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Blood_Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Place_Acquired': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'DRU': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'MuncityOfDRU': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'OnsetToAdmit': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Barangay': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Admitted': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'ClinClass': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'CaseClassification': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Outcome': lambda x: x.mode().iloc[0] if not x.mode().empty else None
        }).reset_index()

        temperature_df['DateMonthYear'] = temperature_df['Date']
        temperature_daily = temperature_df.groupby('DateMonthYear').agg({
            'Max': 'mean', 'Min': 'mean', 'Avg': 'mean'}).reset_index()

        humidity_df['DateMonthYear'] = humidity_df['Date']
        humidity_daily = humidity_df.groupby('DateMonthYear').agg({
            'Specific': 'mean', 'Relative': 'mean', 'Precipitation': 'mean'}).reset_index()

        mosquito_df['DateMonthYear'] = mosquito_df['Trap Date']
        mosquito_daily = mosquito_df.groupby('DateMonthYear').agg({'Count': 'sum'}).reset_index()

        # Merge all data
        all_dates = pd.date_range(start=dengue_df['DateMonthYear'].min(), end=dengue_df['DateMonthYear'].max())
        all_dates_df = pd.DataFrame({'DateMonthYear': all_dates})

        merged_df = all_dates_df.merge(dengue_daily, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(dengue_agg, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(temperature_daily, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(humidity_daily, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(mosquito_daily, on='DateMonthYear', how='left')
        merged_df['MonthYear'] = merged_df['DateMonthYear'].dt.to_period('M').dt.to_timestamp()

        # Fill missing values
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)
        mean_count = merged_df['Count'].mean()
        merged_df['Count'] = merged_df['Count'].fillna(mean_count)

        # Prepare features for prediction
        features = ['Min', 'Specific', 'Precipitation', 'Relative', 'Count', 'Muncity',
            'Blood_Type', 'MuncityOfDRU', 'Barangay', 'ClinClass', 'CaseClassification', 'Place_Acquired']
        
        # Create a DataFrame for prediction
        prediction_df = merged_df[features]

        # Verify the model type
        if hasattr(model, 'predict'):
            try:
                # Predict future cases
                future_month = pd.date_range(start=merged_df['DateMonthYear'].max(), periods=2, freq='M')[1]
                future_dates_df = pd.DataFrame({'DateMonthYear': [future_month] * len(prediction_df)})
                future_dates_df = future_dates_df.merge(prediction_df, left_index=True, right_index=True)

                # Make predictions
                next_month_prediction = model.predict(future_dates_df[features])
                
                # Check the shape of the predictions
                if len(next_month_prediction) != len(future_dates_df):
                    raise ValueError(f"Prediction length ({len(next_month_prediction)}) does not match DataFrame length ({len(future_dates_df)})")

                # Add predictions to the DataFrame
                future_dates_df.loc[:, 'Predicted_Cases'] = next_month_prediction

            except Exception as e:
                flash(f'Prediction error: {e}', 'error')
                return redirect(url_for('predict'))
        else:
            flash('Model loading error. Please check the model file.', 'error')
            return redirect(url_for('predict'))

        # Aggregate predictions by municipality
        municipality_predictions = future_dates_df.groupby('Muncity').agg({
            'Predicted_Cases': 'sum'
        }).reset_index()

        # Ensure all municipalities are included
        all_municipalities = dengue_df['Muncity'].unique()
        all_municipalities_df = pd.DataFrame({'Muncity': all_municipalities})
        municipality_predictions = all_municipalities_df.merge(municipality_predictions, on='Muncity', how='left')
        municipality_predictions['Predicted_Cases'].fillna(0, inplace=True)

        # Create HTML table
        prediction_table = municipality_predictions.to_html(classes='data', header=True, index=False)

        return render_template('predict.html', tables=[prediction_table])

    return render_template('predict.html')


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
     cases_per_municipality_data, _, monthly_admission_plots,
     blood_type_html) = process_data(selected_municipality)

    return render_template('admin.html',
                           month_distribution_html=month_distribution_html,
                           age_gender_html=age_gender_html,
                           cases_per_municipality_html=cases_per_municipality_html,
                           plots=admin_plots,
                           selected_municipality=selected_municipality,
                           cases_per_municipality_data=cases_per_municipality_data,
                           monthly_admission_plots=monthly_admission_plots,
                           blood_type_html=blood_type_html)
    
@app.route('/get_monthly_admission_plots')
def get_monthly_admission_plots():
    year = request.args.get('year')
    if year:
        try:
            # Call process_data to get the plots
            _, _, _, _, _, monthly_admission_plots, _ = process_data()

            # Return the specific plot for the requested year
            plot_html = monthly_admission_plots.get(int(year))
            
            if plot_html:
                return jsonify({'plot_html': plot_html})
            else:
                return jsonify({'error': 'No data available for the selected year.'})
        except Exception as e:
            print(f"Error generating plot: {e}")
            return jsonify({'error': 'Error generating plot.'})
    return jsonify({'error': 'Year not specified'})


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
