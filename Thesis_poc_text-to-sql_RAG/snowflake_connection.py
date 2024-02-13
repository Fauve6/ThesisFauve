import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from snowflake.snowpark import Session


def setup_connection(**kwargs):
    """uses key pair authentication to setup the snowflake connector """
    with open(os.environ["snowflake_private_key_path"], "rb") as key:
        p_key = serialization.load_pem_private_key(
            key.read(),
            password=os.environ['private_key_passphrase'].encode(),
            backend=default_backend()
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())

    connection_parameters = {
        "account": os.environ["snowflake_account"],
        "user": os.environ["snowflake_user"],
        "private_key": pkb,
        # "password": os.environ["snowflake_password"],
        "role": os.environ["snowflake_user_role"],
        "warehouse": os.environ["snowflake_warehouse"],
        "database": os.environ["snowflake_database"],
        "schema": os.environ["snowflake_schema"],
    }

    # update values based on provided kwargs
    for k in kwargs:
        connection_parameters[k] = kwargs[k]

    session = Session.builder.configs(connection_parameters).create()
    sf_settings_overview1 = session.sql("select current_role(), current_user()").collect()
    sf_settings_overview2 = session.sql("select current_warehouse(), current_database(), current_schema()").collect()

    print(sf_settings_overview1)
    print(sf_settings_overview2)

    return session


if __name__ == "__main__":
    print("use explain = True to understand Path ")
    # get_root_path_obj(explain=True)
    print("understood? , turn it off")
