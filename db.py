import psycopg2


def open_database_connection():
    conn = psycopg2.connect("postgres://doni_data_user:yV287EL5Ju5ESwxts8th7A0EPUQfv0Dn@dpg-cpc1pbm3e1ms739dhttg-a"
                            ".oregon-postgres.render.com/doni_data")
    cur = conn.cursor()
    return cur
