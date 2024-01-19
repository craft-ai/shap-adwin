# TODO: def retrain_PH parameters
    def retrain_PH(self, delta_w=0.01, delta_d=0.005, win_size=200, stop_first_detect=False):
        return (self.retrain_detector(detector_func=PageHinkley,
                                      signal="error",
                                      detector_params={
                                          "delta": delta_d, },
                                      warning_params={
                                          "delta": delta_w, },
                                      win_size=win_size,
                                      retrain_name="PH",
                                      stop_first_detect=stop_first_detect))


    def retrain_PH_loss(self, delta_w=0.01, delta_d=0.005, win_size=200, stop_first_detect=False):
        return (self.retrain_detector(detector_func=PageHinkley,
                                      signal="log_loss",
                                      detector_params={"delta": delta_d},
                                      warning_params={"delta": delta_w},
                                      win_size=win_size,
                                      retrain_name="PH_loss",
                                      stop_first_detect=stop_first_detect))

    # TODO: def retrain_PH_point():
    def retrain_PH_point(self, delta_w=0.01, delta_d=0.005, win_size=200, stop_first_detect=False):
        return (self.retrain_detector(detector_func=PageHinkley,
                                      signal="point",
                                      detector_params={"delta": delta_d},
                                      warning_params={"delta": delta_w},
                                      win_size=win_size,
                                      retrain_name="PH_point",
                                      stop_first_detect=stop_first_detect))

    # TODO: def retrain_PH_shap():
    def retrain_PH_shap(self, delta_w=0.01, delta_d=0.005, win_size=200, clock=1, bgd_type="sample",
                        n_samp_bgd=25, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={"delta": delta_d},
                                           warning_params={"delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_logloss_shap",
                                           bgd_type="sample",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))

   def retrain_shap_PH(self, delta_w=0.01, delta_d=0.005, win_size=200, clock=1, bgd_type="sample",
                        n_samp_bgd=25, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={"delta": delta_d},
                                           warning_params={"delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_logloss_shap",
                                           bgd_type="sample",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))
# ------------------------------------------------------------------------------------------------------------------------------------------------

    def retrain_shap_PH_back_worse(self,  delta_w=0.01, delta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):
        """
        shap explainer background data is composed of 50 points with the highest loss

        """
        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d},
                                           warning_params={
                                               "delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="worse",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_PH_back_best(self, delta_w=0.01, delta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):
        """
            background is filled with with the 50 lowest loss points
        """
        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d},
                                           warning_params={
                                               "delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="best",
                                           n_samp_bgd=25,#TODO: pass n_bkg samp as param
                                           stop_first_detect=stop_first_detect))

    def retrain_shap_PH_back_train(self, delta_w=0.01, delta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):
        """
        TODO: check of what to do with duplicate func retrain_shap_PH
        """

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": delta_d},
                                           warning_params={
                                               "delta": delta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="train",
                                           n_samp_bgd=25,
                                           stop_first_detect=stop_first_detect))


# ------------------------------------------------------------------------------------------------------------------------------------------------


    def retrain_shap_PH_smallback(self, beta_w=0.01, beta_d=0.002, win_size=200, return_shap=False, stop_first_detect=False):

        return (self.retrain_detector_shap(detector_func=PageHinkley,
                                           signal="log_loss",
                                           detector_params={
                                               "delta": beta_d},
                                           warning_params={
                                               "delta": beta_w},
                                           win_size=win_size,
                                           retrain_name="PH_loss_shap",
                                           bgd_type="small",
                                           stop_first_detect=stop_first_detect))
