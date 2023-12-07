import pytest
from typing import BinaryIO
from lib.ConfigReader import get_app_config
from lib.Utils import get_spark_session
from lib.DataReader import read_customers,read_orders
from lib.DataManipulation import filter_closed_orders,count_orders_state,filter_orders_generic

def test_read_customers_df():
    spark = get_spark_session("LOCAL")
    customers_count=read_customers(spark,"LOCAL").count()
    assert customers_count==12435

def test_read_orders_df():
    spark = get_spark_session("LOCAL")
    orders_count=read_orders(spark,"LOCAL").count()
    assert orders_count==4
    
      
# @pytest.mark.skip("workinprocess")
# def test_read_customers_df(spark):
#     customers_count=read_customers(spark,"LOCAL").count()
#     assert customers_count==12435
    
# @pytest.mark.skip("workinprocess")
# def test_read_orders_df(spark):
#     orders_count=read_orders(spark,"LOCAL").count()
#     assert orders_count==4
    
@pytest.mark.parametrize("status,count",[("CLOSED",2),("PENDING_PAYMENT",1),("COMPLETE",1)])
@pytest.mark.latest()
def test_check_count_df(spark,status,count):
    orders_df=read_orders(spark,"LOCAL")
    filtered_count=filter_orders_generic(orders_df,status).count()
    assert filtered_count==count
    
@pytest.mark.transformation()
def test_filter_closed_orders(spark):
    orders_df=read_orders(spark,"LOCAL")
    filtered_count=filter_closed_orders(orders_df).count()
    assert filtered_count==2
    
# @pytest.mark.slow()
def test_read_app_config():
    config=get_app_config("LOCAL")
    assert config["orders.file.path"]=="data/orders.csv"

@pytest.mark.skip("work in process")
def test_count_orders_state(spark,expected_results):
    customers_df=read_customers(spark,"LOCAL")
    actual_results=count_orders_state(customers_df)
    assert actual_results.collect()==expected_results.collect()
    
