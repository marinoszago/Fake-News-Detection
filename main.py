import sys, os
from src.Fake_News.Backend.crud_operations.Get.get_all import AllStatements
from src.Fake_News.Backend.crud_operations.Get.get_info import AdditionalInfo
from src.Fake_News.Backend.crud_operations.Get.get_id import StatementsIDs
from src.Fake_News.Frontend.charts.pie_chart import ShowPie_ByAnnotation,createStatementTypePie
from src.Fake_News.Frontend.charts.horizontal_bar import showBarBySubjectCount
from src.Fake_News.Backend.crud_operations.Get.by_ruling_slug import RulingSlug
from src.Fake_News.Backend.crud_operations.Get.get_subjects import Subjects
from src.Fake_News.Backend.database.connection import openConnection,closeConnection
from src.Fake_News.Backend.config.config import *
from src.Fake_News.Backend.crud_operations.Insert.insert_to_collection import Insert
from src.Fake_News.Backend.api.statements.get_subjects import getSubjectsFromApi


# =======================
#      MAIN MENU
# =======================

def main():
    choice = '0'
    while choice == '0':
        print("Main Choice: Choose your choice")
        print("1. All Statements")
        print("2. All Rullings")
        print("3. Pie Chart by Annotation count")
        print("4. Get Subjects from API and store them in DB")
        print("5. Horizontal Bar by Subjects count")
        print("6. Statements by Statement Type")

        choice = input("Please make a choice: ")

        if choice == "6":
            db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
            collection_statements = db[statements_table_name]
            print("OK...Connection Granted")
            ai = AdditionalInfo()
            claims = ai.getStatementsByType(collection_statements, 'Claim')
            attack = ai.getStatementsByType(collection_statements, 'Attack')
            flip = ai.getStatementsByType(collection_statements, 'Flip')

            print("Claim: "+str(claims))
            print("Attack: "+str(attack))
            print("Flip: "+str(flip))

            createStatementTypePie(claims, attack, flip)

        elif choice == "5":
            try:
                db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
                collection_statements = db[statements_table_name]
                collection_statistics_subjects = db[statistics_by_subject]
                # collection_subjects = db[subjects]

                print("OK...Connection Granted")

                """ Get Subject count and store them in DB """

                subject = Subjects()
                # counter = 0
                # sum_of_counter = 0
                # for sub in collection_subjects.find():
                #     print(sub['subject'])
                #     counter = subject.getStatementSubjectCount(collection_statements, sub['subject'])
                #     print("for sub: "+sub['subject']+" count is: "+str(counter))
                #     sum_of_counter = counter + sum_of_counter
                #     data = {"subject": sub['subject'], "count": counter}
                #     collection_statistics_subjects.insert(data)
                #     counter = 0
                # print(sum_of_counter)

                """ Subjects and count & show horizontal bar """
                # print(subject.getSubjectsByCount(collection_statistics_subjects, 500))

                subjects_arr = []
                subjects_count_arr = []

                for sub in subject.getSubjectsByCount(collection_statistics_subjects, 500):
                    subjects_arr.append(sub['subject'])
                    subjects_count_arr.append(sub['count'])
                showBarBySubjectCount(subjects_count_arr, subjects_arr)
            finally:
                print("OK... Closing Connection")
                closeConnection(db.client)
        elif choice == "4":
            """ Get Subjects and store them in mongodb """
            getSubjectsFromApi()
        elif choice == "3":
            try:
                db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
                print("OK...Connection Granted")
                collection_statements = db[statements_table_name]
                """ Get Statement Count by Annotation and save them to MongoDB """
                """ # 1. True
                    # 2. False
                    # 3. pants-fire
                    # 4. full-flop
                    # 5. no-flip
                    # 6. half-flip
                    # 7. barely-true
                    # 8. half-true
                    # 9. mostly-true """

                rs = RulingSlug()

                t_count = rs.getRulingSlagStatementCount(collection_statements, 'true')
                f_count = rs.getRulingSlagStatementCount(collection_statements, 'false')
                pf_count = rs.getRulingSlagStatementCount(collection_statements, 'pants-fire')
                ff_count = rs.getRulingSlagStatementCount(collection_statements, 'full-flop')
                nf_count = rs.getRulingSlagStatementCount(collection_statements, 'no-flip')
                hf_count = rs.getRulingSlagStatementCount(collection_statements, 'half-flip')
                bt_count = rs.getRulingSlagStatementCount(collection_statements, 'barely-true')
                ht_count = rs.getRulingSlagStatementCount(collection_statements, 'half-true')
                mt_count = rs.getRulingSlagStatementCount(collection_statements, 'mostly-true')

                """ Insert statistics to DB """
                # Insert.insert_to_collection(db, statistics_by_annotation, t_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, f_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, pf_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, ff_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, nf_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, hf_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, bt_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, ht_count)
                # Insert.insert_to_collection(db, statistics_by_annotation, mt_count)

                """ Print Counts """
                print(t_count)
                print(f_count)
                print(pf_count)
                print(ff_count)
                print(nf_count)
                print(hf_count)
                print(bt_count)
                print(ht_count)
                print(mt_count)

                """ Show Pie Chart """
                ShowPie_ByAnnotation(
                                     pf_count['Counter'],
                                     ff_count['Counter'],
                                     hf_count['Counter'],
                                     nf_count['Counter'],
                                     f_count['Counter'],
                                     bt_count['Counter'],
                                     ht_count['Counter'],
                                     mt_count['Counter'],
                                     t_count['Counter']
                )
            finally:
                print("OK... Closing Connection")
                closeConnection(db.client)
        elif choice == "2":
            """ ALL RULINGS """
            try:
                db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
                collection_statements = db[statements_table_name]
                print("OK...Connection Granted")
                rulings = AdditionalInfo.getRulings(collection_statements)
                print(rulings)
            finally:
                print("OK... Closing Connection")
                closeConnection(db.client)
        elif choice == "1":
            """ ALL STATEMENTS """
            try:
                db = openConnection(db_hostname, db_name, db_port, db_username, db_password, db_authMechanism)
                collection_statements = db[statements_table_name]
                print("OK...Connection Granted")
                print("OK...Connection Granted")
                statements = AllStatements.getAllStatements(collection_statements)
                print(statements)
            finally:
                print("OK... Closing Connection")
                closeConnection(db.client)
        else:
            print("I don't understand your choice.")


# =======================
#      MAIN PROGRAM
# =======================
if __name__ == "__main__":
    main()

