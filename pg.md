# PostgreSQL Setup on AWS EC2
## 1. Install PostgreSQL
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```
## 2. Start PostgreSQL Service 
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
```
## 3. Configure PostgreSQL for Remote Access
Edit postgresql.conf:
```bash
sudo nano /etc/postgresql/*/main/postgresql.conf
```
Change:
```
listen_addresses = '*'
```
## 4. Configure Authentication
Edit pg_hba.conf:
```bash
sudo nano /etc/postgresql/*/main/pg_hba.conf
```
Add/modify:
```
# Database administrative login by Unix domain socket
local   all             postgres                                md5 
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
host    all             all             0.0.0.0/0               md5
```
## 5. Create Database & Set Password
```bash
# Switch to postgres user
sudo -i -u postgres
# Create database
createdb <DB_NAME>
# Enter psql
psql
# Set password
ALTER USER postgres WITH PASSWORD '<YOUR_PASSWORD>';
\q
# Exit postgres user
exit
```
## 6. Restart PostgreSQL
```bash
sudo systemctl restart postgresql
```
## 7. AWS EC2 Security Group Setup
- Go to EC2 Dashboard → Security Groups
- Add inbound rule:
  - Type: Custom TCP
  - Port: 5432
  - Source: Your IP or 0.0.0.0/0 (for testing only)
  - Description: PostgreSQL access
## 8. Connection Strings
```bash
# Local connection test
psql -U postgres -d <DB_NAME> -W
```
## 9. Verification Commands
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
# Check if postgres is listening on all interfaces
sudo netstat -nlp | grep postgres
```

## Database URLs for Environment Variables
```bash
# For local development (.env.local)
DATABASE_URL=postgresql://postgres:<YOUR_PASSWORD>@localhost:5432/<DB_NAME>

# For production (.env)
DATABASE_URL=postgresql://postgres:<YOUR_PASSWORD>@<EC2_PUBLIC_IP>:5432/<DB_NAME>
```

Note: 
- Replace `<DB_NAME>` with your database name
- Replace `<YOUR_PASSWORD>` with your chosen password
- Replace `<EC2_PUBLIC_IP>` with your actual EC2 public IP
- Using 0.0.0.0/0 in security groups is not recommended for production
