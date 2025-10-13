import pyodbc

server   = "udcserver2025.database.windows.net"
database = "grupo_1"
username = "ugrupo1"
password = "HK9WXIJaBp2Q97haePdY"
driver   = "{ODBC Driver 18 for SQL Server}"

conn = pyodbc.connect(
    f"DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sys.tables;")
for row in cursor.fetchall():
    print(row)

conn.close()
