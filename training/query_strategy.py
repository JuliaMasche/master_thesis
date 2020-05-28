from alipy import ToolBox
import random


def select_query_strategy(alibox, query_str, train_index):

    if query_str == "QueryInstanceGraphDensity":
        al_strategy = alibox.get_query_strategy(strategy_name=query_str, train_idx=train_index)

    if query_str == "QueryInstanceUncertainty":
        al_strategy = alibox.get_query_strategy(strategy_name=query_str)
    
    if query_str == "QueryInstanceRandom":
        al_strategy = alibox.get_query_strategy(strategy_name=query_str)

    if query_str == "QueryInstanceBMDR":
        al_strategy = alibox.get_query_strategy(strategy_name=query_str)

    if query_str == "QueryInstanceSPAL":
        al_strategy = alibox.get_query_strategy(strategy_name=query_str)

    if query_str == "QueryInstanceQBC":
        al_strategy = alibox.get_query_strategy(strategy_name=query_str)

    return al_strategy


def select_next_batch(al_strategy, query_str, label_ind, unlab_ind, batch_size, pred_mat):

    if query_str == "QueryInstanceGraphDensity":
        select_ind = al_strategy.select(label_index=label_ind, unlabel_index=unlab_ind, batch_size=batch_size)

    if query_str == "QueryInstanceUncertainty":
        select_ind = al_strategy.select_by_prediction_mat(list(unlab_ind), pred_mat, batch_size=batch_size)
    
    if query_str == "QueryInstanceRandom":
        select_ind = random.sample(unlab_ind, batch_size)

    if query_str == "QueryInstanceBMDR":
        select_ind = al_strategy.select(label_index=label_ind, unlabel_index=unlab_ind, batch_size=batch_size)

    if query_str == "QueryInstanceSPAL":
        select_ind = al_strategy.select(label_index=label_ind, unlabel_index=unlab_ind, batch_size=batch_size)

    if query_str == "QueryInstanceQBC":
        select_ind = al_strategy.select_by_prediction_mat(list(unlab_ind), pred_mat, batch_size=batch_size)

    return select_ind



        
