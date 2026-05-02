# Event Management Dashboard

## Project Overview
This project is a full-stack **Event Management Dashboard** developed for university coursework.  
It allows users to log in with different roles and interact with the system based on their permissions.

The system includes:
- a **FastAPI backend**
- a **Bootstrap frontend**
- a **SQLite database**
- **machine learning prediction**
- **real client data integration** for footfall and survey analysis

The dashboard is designed to manage events, analyse client data, and demonstrate role-based access control in a realistic university event setting.

---

## Main Features

### User Authentication and Security
- Secure login system
- Password hashing using **bcrypt**
- Session-based token authentication
- Logout support

### Role-Based Access Control
The system supports 3 user roles:

#### Admin
- Log in to the dashboard
- View all users
- Create users
- Change user roles
- Delete users
- View managed events
- Create managed events
- Delete managed events
- View real client event data
- View survey analytics
- Use ML prediction

#### Staff
- Log in to the dashboard
- View managed events
- Create managed events
- View real client event data
- Use ML prediction
- Search real client event data

#### Viewer
- Log in to the dashboard
- View managed events
- View real client event data
- Search real client event data
- Cannot create events
- Cannot delete events
- Cannot manage users
- Cannot use ML prediction

---

## Data and Analytics Features

### Managed Events
Users with permission can create and manage events stored in the SQLite database.

### Client Footfall Data
Real footfall data is imported from the client CSV file and stored in SQLite.  
This data is used for:
- client event views
- search and filtering
- machine learning prediction

### Survey Analytics
Survey data is imported from the client Excel file and analysed in the dashboard.  
The system shows:
- total survey responses
- feedback category chart
- suggestion chart
- summary of most common responses

### Machine Learning
A **Linear Regression** model is trained using the real client footfall dataset.

The model predicts footfall using:
- Total Visiting
- Total Passing Through

The system also classifies predicted results into:
- Low
- Medium
- High

### Smart Search
Users can search real client event data using:
- event name
- area / location
- date
- minimum footfall
- maximum footfall
- whether an event took place

---

## Technologies Used
- **Python**
- **FastAPI**
- **SQLite**
- **Pandas**
- **Scikit-learn**
- **bcrypt**
- **HTML**
- **Bootstrap**
- **JavaScript**
- **Chart.js**

---

## Project Structure

```text
Software/
│
├── main.py
├── events.db
├── footfall_data.csv
├── survey_data.xlsx
├── requirements.txt
├── README.md
│
└── static/
    └── dashboard.html