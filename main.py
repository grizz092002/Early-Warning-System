import pandas as pd
import json
import os
import warnings
import sklearn
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import firebase_admin
import requests
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from firebase_admin import credentials, auth, initialize_app, firestore, storage
from sklearn.svm import SVC
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session, flash
from dotenv import load_dotenv
from datetime import datetime, timedelta
from data_processing import process_data
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from flask_mail import Mail, Message
from dateutil import parser
from twilio.rest import Client

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_secret_key'  

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'josephcollantes65@gmail.com'  # Replace with your Gmail address
app.config['MAIL_PASSWORD'] = 'oewu ulom kpax koqr'       # Replace with your Gmail App Password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

# Initialize Twilio client
account_sid = 'ACe3f45dff58eb11d6e018fc205bffc16b'
auth_token = '28ea2a902fb6d439254589319017f19a'
twilio_phone_number = '09071103861'  # The phone number you bought from Twilio

client = Client(account_sid, auth_token)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/auth-key.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dengue-prediction-d862e.appspot.com'  # Replace with your Firebase Storage bucket URL
})
db = firestore.client()
bucket = storage.bucket()

mail = Mail(app)

load_dotenv()

UPLOAD_FOLDER = 'uploads'
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
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError('Unsupported file format')
    
def upload_to_firebase(file, filename, folder_name):
    file.seek(0)  # Reset file stream to the beginning
    blob = bucket.blob(f'{folder_name}/{filename}')
    blob.upload_from_file(file, content_type=file.content_type)
    return blob.public_url
    
# Set cache control headers
def set_no_cache(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Load the pre-trained model
model_path = 'C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/best_gb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Process uploaded files
        files = {
            'file1': 'temperature',
            'file2': 'dengue',
            'file3': 'humidity',
            'file4': 'mosquito'
        }
        dfs = {}
        expected_columns = {
            'temperature': ['Date', 'Max', 'Min', 'Avg'],
            'dengue': ['DateOfEntry', 'Muncity', 'Sex', 'Blood_Type', 'Place_Acquired', 'DRU', 'MuncityOfDRU', 'OnsetToAdmit', 'Barangay', 'Admitted', 'Type', 'ClinClass', 'CaseClassification', 'Outcome'],
            'humidity': ['Date', 'Specific', 'Relative', 'Precipitation'],
            'mosquito': ['Trap Date', 'Genus', 'Specific Epithet', 'Gender', 'Count']
        }

        for file_key, file_type in files.items():
            if file_key in request.files:
                file = request.files[file_key]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Upload to Firebase with folder
                    public_url = upload_to_firebase(file, filename, file_type)
                    print(f"File uploaded to Firebase Storage: {public_url}")
                    
                    try:
                        df = read_file(file_path)
                    except ValueError as e:
                        flash(f'File processing error: {e}', 'error')
                        return redirect(url_for('predict'))
                    
                    if not all(col in df.columns for col in expected_columns[file_type]):
                        flash(f'File "{filename}" is missing required columns for {file_type}.', 'error')
                        return redirect(url_for('predict'))
                    dfs[file_key] = df

        # Check if all required files are uploaded
        if len(dfs) != len(files):
            flash('Please upload all required files.', 'error')
            return redirect(url_for('predict'))

        # Data processing and merging
        dengue_df = dfs['file2']
        temperature_df = dfs['file1']
        humidity_df = dfs['file3']
        mosquito_df = dfs['file4']
        
        # Convert date columns to datetime with the correct format
        dengue_df['DateOfEntry'] = pd.to_datetime(dengue_df['DateOfEntry'], format='%d/%m/%Y')
        temperature_df['Date'] = pd.to_datetime(temperature_df['Date'], format='%d/%m/%Y')
        mosquito_df['Trap Date'] = pd.to_datetime(mosquito_df['Trap Date'], format='%d/%m/%Y')
        humidity_df['Date'] = pd.to_datetime(humidity_df['Date'], format='%d/%m/%Y')

        dengue_df['Age'] = dengue_df['AgeYears'] + dengue_df['AgeMons'] / 12 + dengue_df['AgeDays'] / 365
        dengue_df['DateMonthYear'] = dengue_df['DateOfEntry'].dt.to_period('D').dt.to_timestamp()

        dengue_daily = dengue_df.groupby('DateMonthYear').size().reset_index(name='Cases')
        dengue_agg = dengue_df.groupby('DateMonthYear').agg({
            'Muncity': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'AdmitToEntry': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'OnsetToAdmit': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'LabRes': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'MorbidityMonth': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'MorbidityWeek': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Age': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
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
        mosquito_daily = mosquito_df.groupby('DateMonthYear').agg({
            'Genus': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Gender': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Specific Epithet': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Count': 'sum'}).reset_index()

        all_dates = pd.date_range(start=dengue_daily['DateMonthYear'].min(), end=dengue_daily['DateMonthYear'].max())
        all_dates_df = pd.DataFrame({'DateMonthYear': all_dates})

        merged_df = all_dates_df.merge(dengue_daily, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(dengue_agg, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(temperature_daily, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(humidity_daily, on='DateMonthYear', how='left')
        merged_df = merged_df.merge(mosquito_daily, on='DateMonthYear', how='left')
        merged_df['MonthYear'] = merged_df['DateMonthYear'].dt.to_period('M').dt.to_timestamp()
        
        # Define categorical columns
        categorical_cols = ['Muncity', 'Sex', 'Blood_Type', 'Place_Acquired', 'Barangay',
                            'ClinClass', 'CaseClassification', 'Type', 'LabRes']
        
        fb_fill = [
            'Genus', 'Gender', 'Specific Epithet', 'Count'
        ]

        # Forward fill for specified columns
        merged_df_ffill = merged_df.copy()  # Make a copy to avoid modifying the original DataFrame
        merged_df_ffill[fb_fill] = merged_df_ffill[fb_fill].ffill()

        # Backward fill for specified columns
        merged_df_fill = merged_df_ffill.copy()  # Make a copy to avoid modifying the forward-filled DataFrame
        merged_df_fill[fb_fill] = merged_df_fill[fb_fill].bfill()

        # List of columns to fill with 0
        zero_fill = [
            'Cases', 'Muncity', 'AdmitToEntry', 'OnsetToAdmit', 'LabRes',
            'MorbidityMonth', 'MorbidityWeek', 'Age', 'Sex', 'Blood_Type',
            'Place_Acquired', 'DRU', 'MuncityOfDRU', 'Barangay', 'Admitted',
            'Type', 'ClinClass', 'CaseClassification', 'Outcome'
        ]

        # Fill NaN values with 0 for the specified columns
        merged_df_fill[zero_fill] = merged_df_fill[zero_fill].fillna(0)
        
        print(merged_df_fill.dtypes)
        
                # Convert categorical columns to string type to avoid mixed types
        for col in categorical_cols:
            merged_df_fill[col] = merged_df_fill[col].astype(str)
    
        # Define your features and categorical columns
        features = ['Count', 'Muncity', 'Sex', 'Age', 'Blood_Type', 
                    'Place_Acquired', 'Barangay', 'ClinClass', 'CaseClassification',
                    'Type', 'LabRes', 'MorbidityMonth', 'Admitted', 'MorbidityWeek']

        # Prepare X_new with only the selected features
        X_new = merged_df_fill[features]

        # Create LabelEncoder instances
        label_encoders = {col: LabelEncoder() for col in categorical_cols}

        # Apply LabelEncoder to each categorical column
        for col in categorical_cols:
            X_new[col] = label_encoders[col].fit_transform(X_new[col])

        # Ensure X_new has exactly 14 features
        if X_new.shape[1] != 14:
            raise ValueError(f"Expected 14 features, but got {X_new.shape[1]}")

        # Predict the dengue cases for the next year
        merged_df_fill['Predicted_Cases'] = model.predict(X_new)

        # Shift the dates to predict for the next year
        merged_df_fill['DateMonthYear'] = merged_df_fill['DateMonthYear'] + pd.DateOffset(years=1)

        # Filter only for the next year
        next_year = merged_df_fill['DateMonthYear'].dt.year.max()
        predictions_next_year = merged_df_fill[merged_df_fill['DateMonthYear'].dt.year == next_year]

        # Aggregate predictions by municipality and month
        municipality_monthly_predictions = predictions_next_year.groupby(['DateMonthYear', 'Muncity']).agg({
            'Predicted_Cases': 'sum'
        }).reset_index()

        # Ensure all municipalities are included
        all_municipalities = dengue_df['Muncity'].unique()
        all_municipalities_df = pd.DataFrame({'Muncity': all_municipalities})
        municipality_monthly_predictions = municipality_monthly_predictions.merge(all_municipalities_df, on='Muncity', how='right')
        municipality_monthly_predictions['Predicted_Cases'] = municipality_monthly_predictions['Predicted_Cases'].fillna(0).astype(int)  # Convert to integer

        # Convert predictions to a list of dictionaries for easy rendering
        predictions = municipality_monthly_predictions.to_dict(orient='records')

        return render_template('predict.html', predictions=predictions)
    
    return render_template('predict.html')


@app.route('/contact_numbers', methods=['GET'])
def get_contact_numbers():
    # Initialize Firestore client
    db = firestore.client()

    # Reference to the contact numbers document
    contact_numbers_ref = db.collection('admin_acc').document('contact_numbers')
    contact_numbers_doc = contact_numbers_ref.get()

    if contact_numbers_doc.exists:
        contact_numbers = contact_numbers_doc.to_dict()
        return jsonify(contact_numbers)
    else:
        return jsonify({'error': 'Contact numbers not found'}), 404
    
@app.route('/update-contact-numbers', methods=['POST'])
def update_contact_numbers():
    data = request.json
    print('Received data:', data)  # Debugging output
    try:
        # Initialize Firestore client
        db = firestore.client()

        # Reference to the contact numbers document
        contact_numbers_ref = db.collection('admin_acc').document('contact_numbers')

        # Get the existing document data
        existing_data = contact_numbers_ref.get().to_dict() or {}

        # Update the document with the new contact numbers
        existing_data.update(data)
        contact_numbers_ref.set(existing_data)
        return jsonify({'success': True}), 200
    except Exception as e:
        print('Error updating contact numbers:', e)  # Debugging output
        return jsonify({'success': False}), 500


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        municipality = request.form['municipality']
        contact_number = request.form['contact_number']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        try:
            db.collection('user_acc').add({
                'fullname': fullname,
                'municipality': municipality,
                'contact_number': contact_number,
                'email': email,
                'password': hashed_password,
                'status': 'pending'
            })
            flash('Account registered successfully! Please wait for admin approval.')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Error: {e}")
            flash('Database operation failed. Please try again later.')
            return redirect(url_for('register'))

    return render_template('registration.html')

@app.after_request
def after_request(response):
    if request.endpoint in ['user_page', 'admin_page']:
        set_no_cache(response)
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            # Check in admin_acc collection first
            admin_ref = db.collection('admin_acc').where('username', '==', username).stream()
            admin = next(admin_ref, None)

            if admin:
                admin_data = admin.to_dict()
                if admin_data['password'] == password:  # Plain text comparison
                    session['admin'] = username  # Set session for admin
                    return redirect(url_for('admin_page'))

            # Check in user_acc collection if not an admin
            user_ref = db.collection('user_acc').where('email', '==', username).stream()
            user = next(user_ref, None)

            if user:
                user_data = user.to_dict()
                if check_password_hash(user_data['password'], password):
                    if user_data['status'] == 'approved':
                        session['user'] = username  # Set session for user
                        return redirect(url_for('user_page'))
                    else:
                        flash('Your account is not approved. Please contact support.')
                        return redirect(url_for('login'))

            flash('Invalid username or password')
            return redirect(url_for('login'))

        except Exception as e:
            print(f"Error: {e}")
            flash('Database connection failed')
            return redirect(url_for('login'))

    error = request.args.get('error')
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('admin', None)
    return redirect(url_for('login'))



@app.route('/user')
def user_page():
    if 'user' not in session:
        return redirect(url_for('login'))

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
    if 'admin' not in session:
        return redirect(url_for('login'))

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
    
@app.route('/api/users', methods=['GET'])
def get_users():
    users_ref = db.collection('user_acc')
    docs = users_ref.stream()

    users = []
    for doc in docs:
        user = doc.to_dict()
        user['id'] = doc.id  # Include document ID if needed
        users.append(user)

    return jsonify(users)
            
@app.route('/api/approve_user', methods=['POST'])
def approve_user():
    data = request.json
    email = data.get('email')
    fullname = data.get('fullname')
    password = data.get('password')

    if not email:
        return jsonify({'error': 'No email provided'}), 400

    try:
        # Update status in user_acc collection
        user_ref = db.collection('user_acc').where('email', '==', email).limit(1).get()
        
        if not user_ref:
            return jsonify({'error': 'User not found'}), 404
        
        user_doc = user_ref[0].reference
        user_doc.update({'status': 'approved'})
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in approve_user route: {e}")
        return jsonify({'error': f'Database operation failed: {str(e)}'}), 500


@app.route('/api/reject_user', methods=['POST'])
def reject_user():
    try:
        data = request.json
        email = data.get('email')

        if not email:
            return jsonify({'status': 'error', 'error': 'No email provided'}), 400

        # Reference to the Firestore collection
        users_ref = db.collection('user_acc')
        query = users_ref.where('email', '==', email).limit(1).get()

        if query:
            doc_id = query[0].id
            user_ref = users_ref.document(doc_id)
            user_ref.update({'status': 'rejected'})
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'error': 'No user found with that email'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/approve_accounts')
def approve_accounts():
    return render_template('approve_accounts.html')

@app.route('/repository', methods=['GET'])
def repository():
    # List of files in Firebase Storage
    blobs = bucket.list_blobs()
    
    # Collect folder and file details
    folders = {}
    for blob in blobs:
        folder_name = blob.name.split('/')[0]  # Assuming folder structure
        file_info = {
            'name': blob.name,
            'url': blob.public_url
        }
        if folder_name not in folders:
            folders[folder_name] = []
        folders[folder_name].append(file_info)
    
    # Convert to list of folders with files
    folder_list = [{'name': name, 'files': files} for name, files in folders.items()]
    
    return render_template('repository.html', folders=folder_list)

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/feedbacks')
def feedbacks():
    return render_template('feedbacks.html')

@app.route('/messages')
def messages():
    return render_template('messages.html')

@app.route('/forgotpass', methods=['GET', 'POST'])
def forgotpass():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        print(f'Attempting to find email: {email}')  # Debug log

        try:
            # Query to find documents where the email field matches
            users = db.collection('user_acc').where('email', '==', email).get()

            # Check if the query returned any documents
            if len(users) == 0:
                return render_template('forgotpass.html', error='Email not found.')

            # Generate a password reset link
            try:
                reset_link = auth.generate_password_reset_link(email)
            except Exception as e:
                print(f'Error generating password reset link: {e}')
                return render_template('forgotpass.html', error='Failed to generate password reset link.')

            # Send email
            try:
                msg = Message('Password Reset Request',
                              sender='your-gmail-address@gmail.com',
                              recipients=[email])
                msg.body = f'Please click the following link to reset your password: {reset_link}'
                mail.send(msg)
            except Exception as e:
                print(f'Error sending email: {e}')
                return render_template('forgotpass.html', error='Failed to send email.')

            return render_template('forgotpass.html', success='An email with password reset instructions has been sent to your email address.')
        except Exception as e:
            print(f'General error: {e}')
            return render_template('forgotpass.html', error='There was an error processing your request.')
    else:
        return render_template('forgotpass.html')

@app.route('/resetpass', methods=['GET', 'POST'])
def resetpass():
    if request.method == 'POST':
        oob_code = request.form.get('oobCode')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if new_password != confirm_password:
            return render_template('resetpass.html', error='Passwords do not match.')

        try:
            # Confirm the password reset code and update the password
            auth.confirm_password_reset(oob_code, new_password)
            return render_template('resetpass.html', success='Your password has been successfully reset.')
        except Exception as e:
            print(f'Error resetting password: {e}')
            return render_template('resetpass.html', error='Error resetting password. Please try again.')
    else:
        oob_code = request.args.get('oobCode')
        if not oob_code:
            return redirect(url_for('forgotpass'))
        return render_template('resetpass.html', oobCode=oob_code)

@app.route('/')
def home():
    error = request.args.get('error')
    
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

    return render_template('home.html', error=error,
                           month_distribution_html=month_distribution_html,
                           age_gender_html=age_gender_html,
                           cases_per_municipality_html=cases_per_municipality_html,
                           plots=user_plots,
                           selected_municipality=selected_municipality,
                           cases_per_municipality_data=cases_per_municipality_data,
                           monthly_admission_pivot=monthly_admission_pivot,
                           blood_type_html=blood_type_html)
    
@app.route('/send-alerts', methods=['POST'])
def send_alerts():
    data = request.json
    try:
        for municipality, contact_number in data.items():
            message = f"Alert: Dengue prediction results are available for {municipality}."
            # Send SMS
            client.messages.create(
                body=message,
                from_=twilio_phone_number,  # Your Twilio phone number
                to=contact_number
            )
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(e)
        return jsonify({'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True)
