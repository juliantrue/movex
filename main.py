from fig import Config as C

from evaluate import run_mot16_eval, run_mot16_trace


def main():
    print("App configuration:")
    C.show()
    config_as_dict = C.as_dict()
    app_config = config_as_dict["app"]
    app = list(app_config.keys())[0]

    if len(app_config) > 1:
        raise Exception(
            f"Multiple apps specified: {list(app_config.keys())}. "
            + "Only one type of app should be specified."
        )

    elif app == "mot16_eval":
        run_mot16_eval(C.app_mot16_eval_path_to_mot_dir, C.app_mot16_eval_results_dir)

    elif app == "mot16_trace":
        run_mot16_trace(
            C.app_mot16_trace_path_to_mot_trace_dir, C.app_mot16_trace_results_dir
        )

    else:
        raise Exception(
            f"{app} is not in list of available applications: "
            + '["mot16_eval", mot16_trace]'
        )


if __name__ == "__main__":
    main()
