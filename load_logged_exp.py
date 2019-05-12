import pickle



# Loading data from the folder below
    
# Full version
adult_OLS = pickle.load(open('logged_exp/adult_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.17, 0.2, 0.23, 0.255, 0.265, 0.27, 0.275, 0.31, 1]OLS.pkl', 'rb'))
# adult_XGB = pickle.load(open('adult_full_XGBClassifier_[0.04,0.06,0.08].pkl', 'rb'))
# adult_XGB = pickle.load(open('logged_exp/adult_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 1]_XGB.pkl', 'rb'))
adult_Logistic = pickle.load(open('logged_exp/adult_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 1]_Logistic.pkl', 'rb'))
adult_bl = pickle.load(open('logged_exp/adult_benchmarks.pkl', 'rb'))
adult_RF = pickle.load(open('logged_exp/adult_full_RF.pkl', 'rb'))
adult_LS_XGB = pickle.load(open('logged_exp/adult_LS_tree.pkl', 'rb'))
# adult_LR_XGB = pickle.load(open('adult_full_XGBClassifier_[0.04,0.06,0.08].pkl', 'rb'))
adult_XGB = pickle.load(open('logged_exp/adult_LR_XGB_newrun.pkl', 'rb'))



# disp_curve_list([adult_OLS, adult_XGB, adult_Logistic, adult_RF], adult_bl)

lawschool_XGB =  pickle.load(open('logged_exp/law_school_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.34, 0.35, 0.36, 0.37, 0.38, 0.42, 1]_XGB.pkl', 'rb'))
lawschool_RF = pickle.load(open('logged_exp/law_school_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 1]_RF.pkl', 'rb'))
lawschool_OLS = pickle.load(open('logged_exp/law_school_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.42, 1]_OLS.pkl', 'rb'))
lawschool_bl = pickle.load(open('logged_exp/law_school_full_bl.pkl', 'rb'))
# disp_curve_list([lawschool_XGB, lawschool_RF, lawschool_OLS], lawschool_bl)

comm_OLS = pickle.load(open('logged_exp/communities_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 1]square_DPOLS.pkl', 'rb'))
comm_XGB = pickle.load(open('logged_exp/communities_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 1]square_DPXGB.pkl', 'rb'))
comm_RF = pickle.load(open('logged_exp/communities_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 1]square_DPRF.pkl', 'rb'))
comm_SVM = pickle.load(open('logged_exp/communities_full_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]SVM.pkl', 'rb'))
comm_bl = pickle.load(open('logged_exp/comm_full_bl.pkl', 'rb'))
    # disp_curve_list([comm_RF, comm_OLS, comm_SVM], comm_bl)

# Short version with subsampling
adult_short_OLS = pickle.load(open('logged_exp/adult_short_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.32, 0.34, 0.4, 1]_OLS.pkl', 'rb'))
adult_short_Logistic = pickle.load(open('logged_exp/adult_small_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.32, 0.34, 0.4, 1]LogisticC10.pkl', 'rb'))
adult_short_Logistic2 = pickle.load(open('logged_exp/adult_short_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.32, 0.34, 0.4, 1]_Logistic.pkl', 'rb'))
adult_short_SVM = pickle.load(open('logged_exp/adult_short_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.32, 0.34, 0.4, 1]_SVM.pkl', 'rb'))
adult_short_bl = pickle.load(open('logged_exp/adult_short_bl.pkl', 'rb'))
# disp_curve_list([adult_short_SVM, adult_short_OLS, adult_short_Logistic], adult_short_bl)

lawschool_short_OLS = pickle.load(open('logged_exp/law_school_short_eps_list_[0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.32, 1]OLS.pkl', 'rb'))
lawschool_short_SVM = pickle.load(open('logged_exp/law_school_short_svm.pkl', 'rb'))
lawschool_short_bl = pickle.load(open('logged_exp/law_school_short_bl.pkl', 'rb'))
# disp_curve_list([lawschool_short_OLS, lawschool_short_SVM], lawschool_short_bl)


# Data for number of calls
adult_Ncalls_Logistic = pickle.load(open('logged_exp/adult_short_eps_list_[0.01, 0.04, 0.08, 0.12, 0.15, 0.2, 0.25, 1]Logistic_N_Calls.pkl', 'rb'))
adult_Ncalls_OLS = pickle.load(open('logged_exp/adult_short_eps_list_[0.01, 0.04, 0.08, 0.12, 0.15, 0.2, 0.25, 1]OLS_N_Calls.pkl', 'rb'))
adult_Ncalls_SVM = pickle.load(open('logged_exp/adult_short_eps_list_[0.01, 0.04, 0.08, 0.12, 0.15, 0.2, 0.25, 1]SVM_N_Calls.pkl', 'rb'))

lawschool_Ncalls_OLS = pickle.load(open('logged_exp/law_school_short_eps_list_[0.01, 0.025, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.4, 1]OLS_NCalls.pkl', 'rb'))
lawschool_Ncalls_SVM = pickle.load(open('logged_exp/law_school_short_eps_list_[0.01, 0.025, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.4, 1]SVM_NCalls.pkl', 'rb'))

comm_Ncalls_OLS = pickle.load(open('logged_exp/communities_full_eps_list_[0.01, 0.025, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]OLS_NCalls.pkl', 'rb'))
comm_Ncalls_SVM = pickle.load(open('logged_exp/communities_full_eps_list_[0.01, 0.025, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]SVM_NCalls.pkl', 'rb'))

# benchmark with the fair classification algo with logistic regression oracle
adult_FC_tree = pickle.load(open('adult_grid_tree.pkl', 'rb'))
adult_FC_lin = pickle.load(open('adult_grid_lin.pkl', 'rb'))
adult_short_FC_lin = pickle.load(open('adult_short_FC_lin.pkl', 'rb'))

