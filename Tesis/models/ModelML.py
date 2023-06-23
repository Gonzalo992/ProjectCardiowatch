from .entities.Ml import Ml
from werkzeug.security import generate_password_hash

class ModelML():

    @classmethod
    def login(self, db, ml):
        try:
            cursor = db.connection.cursor()
            sql = """SELECT id, anio, enfermedad, distrito, PDQ FROM modelo 
                    WHERE anio = '{}' and enfermedad = '{}' and distrito = '{}'""".format(ml.anio,ml.enfermedad,ml.distrito)
            cursor.execute(sql)
            row = cursor.fetchone()
            if row != None:
                ml = Ml(row[0], row[1], row[2], row[3], row[4])
                return ml
            else:
                return None
        except Exception as ex:
            raise Exception(ex)

    @classmethod
    def register(self, db, ml):
        try:
            cursor = db.connection.cursor()
            sql = """INSERT INTO modelo(anio,enfermedad,distrito,PDQ) VALUES('{}','{}','{}','{}')""".format(ml.anio,ml.enfermedad,ml.distrito,ml.PDQ)
            cursor.execute(sql)
            '''
            row = cursor.fetchone()
            if row != None:
                user = User(row[0], row[1], User.check_password(row[2], user.password), row[3])
                return user
            else:
                return None
            '''
        except Exception as ex:
            raise Exception(ex)