## Introduction of Peicheng Gu
As a graduate student in computer science from Saint Louis University, I am deeply passionate about the field of computer science and actively seek opportunities for continuous learning and growth. I invite you to visit my personal website, where you can find detailed information about me and insights into some of the projects I have been involved in.

### Food-Online-Ordering-System

This is a Django-based framework to develop a web page that allows users to order food online and check the status of the order. This tool helps customers to order food better, and businesses can also more easily show customers the food information. The core technologies and key points of this project include:

- The code 'manage.py' is a key component of the Django project and is used to manage the various tasks and operations of the project. It shows you how to interact with a Django project from the command line.
- 'admin.py'code is used to customize the look and behavior of Django's admin background. It includes defining custom administrative classes to control the display and permissions of models, as well as registering these models for administration in the administrative background. You also personalize the look of the admin backend by setting titles and taglines.

- In the 'models.py' file, it defines four model classes: Product, Contact, Orders, and OrderUpdate, which describe the data structures in the application. Allows you to store and retrieve information about products, contact information, orders, and order updates in the database.
- The code 'urls.py' is the URL configuration in the Django project. These URL configurations define routes for different pages in the application, allowing users to access different functions and views by accessing different urls.

- The 'views.py' code imports some of the necessary modules and libraries, including model classes, message processing, authentication, payment, and so on. These view functions process requests according to different URL routing, thus achieving different page rendering and interaction functions.

- The code '0001_initial' defines four data models: Contact, Orders, OrderUpdate, and Product. These models define the table structures and fields in the database, and you can use Djangos data migration tool to create database tables and store the data of these models in the database.
- 'index.html' is a simple welcome page that uses the styles and components provided by Bootstrap to render the content, and includes some JavaScript libraries for interactive page effects. Users can click buttons to access different entry pages, including customer, administrator, and employee entry.
- The Python code 'Checksum.py' is an example of integration with the Paytm payment gateway to generate and validate checksums for payment requests. Checksum is a technique used to verify data integrity and is often used in payment transactions to ensure the security and integrity of transaction requests.

### Travel Search System

This is a web page based on Django and MySQL, mainly for visitors to find and search tourist attractions, restaurants and hotel information. The core technologies and key points of this project include:

- The code 'manage.py' is used to configure environment variables for a Django project and to perform various administrative tasks through command-line administration tools. It also includes some exception handling to provide friendly error messages.

- The purpose of this '_init_.py' code is to configure pymysql as the engine for the MySQL database in a Django project so that the project can use pymysql to connect to and manipulate the MySQL database.

- These settings in 'Settings.py' are the core configuration of a Django project and are used to define the behavior and features of the project.

- The 'models.py' code defines three models, TourPlace, Restaurant, and Hotel. These models are used to represent different entities in the database and to create database tables by inheriting the models.model class.
- The 'urls.py' code defines the URL routing rules for the Django project. These URL routing rules define the URL paths for the individual pages in the project and the view functions that should be executed when those paths are requested. For example, the '/tourist' path maps requests to a view function named views.tourist. Similarly, the '/restaurant' path maps the request to a view function named views.restaurant
- The 'views.py' code contains various view functions that handle requests for different pages and retrieve and render data from the database based on the parameters in the request.
- What '0001_initial.py' does is that when the database migration command is executed, Django creates or updates the database table structure based on the actions in the migration file to reflect the changes in the application model. This keeps the application's database consistent with the model, while allowing database schema evolution.



### Analysis Of US Stock Trading Data

This tool can visually show the highest and lowest prices of stocks in a period of time and the rise and fall range, helping
investors to better grasp the stock dynamics. The core technologies and key points of this project include:

- The 'stock_ui_v1.py' code is the user interface part of a GUI application created using PyQt5, which contains a variety of Qt widgets and layout managers, as well as some custom visual styles. By using these techniques, you can build interactive desktop applications.

- The code 'stock_local_analysis.py' is used to calculate and analyze financial data. The following are the main functions of the code:
  1. Load financial data into Pandas DataFrame by reading Excel files.
  2. Calculate and generate summary data, including number of transactions per second, turnover, up and down turnover, VWAP, etc.
  3. According to the time point T entered by the user and the observation period N seconds, calculate the time points in which the stock price rises and falls by 0.5%, 1%, 1.5% and 2% in the time period after time T, and save the results in an Excel file.
  4. Calculate the maximum increase, maximum decline and maximum range of the stock price during the observation period, and save the results in an Excel file.

- The code 'stock_gui.py' uses a stock visualization application created by PyQt5 and Matplotlib. The main function of this application is to visualize stock prices and transaction data, and allow users to calculate the rise and fall.

- The code 'business_logic.py' is used to process local stock data and provide some basic data analysis capabilities.

- The main function of the 'chart.py' code is to create a dynamic VWAP graph that updates dynamically as data points are added.

- Game development: Go's support for low-level memory manipulation and cross-compiling capabilities make it a good choice for game development
- The code 'data.py' first uses Pandas to read data from an Excel file named 'data.xlsx'. The first 120 rows of data were then selected from the data read, and the data for the 'HMS', 'pen/second', and 'VWAP' columns were extracted.
- 'lines.py' This code defines a function create_chart that draws a dynamic double line chart, which can be used to display the volume and VWAP data on the timeline.








