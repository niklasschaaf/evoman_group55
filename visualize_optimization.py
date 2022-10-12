import optuna
import plotly

filename = "optuna_2022-10-12 08:14:06.400095"
study_name = "tuning/"+filename  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(directions=["maximize"], study_name=study_name, storage=storage_name, load_if_exists=True)

fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_contour(study)
fig.show()

print(study.best_params)

#print(study.best_params)
# # plot pareto front
# fig = optuna.visualization.plot_pareto_front(study, target_names=["average fitness gain", "maximum fitness gain"])
# fig.show()
#
# fig = optuna.visualization.plot_optimization_history(study,  target=lambda t: t.values[0], target_name = "average fitness gain")
# fig.show()
#
# fig = optuna.visualization.plot_optimization_history(study,  target=lambda t: t.values[1], target_name = "maximum fitness gain")
# fig.show()
#
# # doesn't work for some reason
# # fig = optuna.visualization.plot_intermediate_values(study)
# # fig.show()
#
# fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name = "average fitness gain")
# fig.show()
#
# fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name = "maximum fitness gain")
# fig.show()
# # more visualization options can be found on optuna.readthedocs.io
