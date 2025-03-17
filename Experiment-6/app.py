from flask import Flask, render_template, request, redirect, url_for, flash, session
import mysql.connector
import random

app = Flask(__name__)
app.secret_key = 'abc'  

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',  
        password='6968',  
        database='adlab'
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    name = request.form['name']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (name, password) VALUES (%s, %s)", (name, password))
        conn.commit()
        flash('Sign up successful! You can now log in.', 'success')
    except mysql.connector.Error as err:
        flash('Error: {}'.format(err), 'danger')
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE name = %s AND password = %s", (name, password))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        # Store user info in session
        session['user_id'] = user[0]  # Assuming first column is id
        session['name'] = user[1]     # Assuming second column is name
        
        # Redirect to dashboard instead of showing message directly
        return redirect(url_for('dashboard'))
    else:
        flash('Login failed! Invalid credentials.', 'danger')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('index'))
    
    # Generate random marks
    marks = random.randint(50, 100)
    grade = 'A' if marks >= 85 else 'B' if marks >= 70 else 'C'
    
    return render_template('dashboard.html', name=session['name'], marks=marks, grade=grade)

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        # Validate that new password and confirm password match
        if new_password != confirm_password:
            flash('New password and confirm password do not match', 'danger')
            return redirect(url_for('change_password'))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if current password is correct
        cursor.execute("SELECT * FROM users WHERE id = %s AND password = %s", 
                      (session['user_id'], current_password))
        user = cursor.fetchone()
        
        if user:
            # Update the password
            cursor.execute("UPDATE users SET password = %s WHERE id = %s", 
                          (new_password, session['user_id']))
            conn.commit()
            flash('Password changed successfully', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Current password is incorrect', 'danger')
        
        cursor.close()
        conn.close()
    
    return render_template('change_password.html')

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)