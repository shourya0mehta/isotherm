"""Every test starts without the machine's real settings and leaves none behind. The command line reads `.env` from
the working folder without overriding what is set (wxbot/__main__.py), and on the Mac or the server `.env` holds the
phone-alert topic and, once live, the real Kalshi key: a test that runs the command line from the repo folder used to
leave those in the environment for every later test (seen on the AWS server, Oct 6: an alert test found a topic)."""
import pytest

REAL_SETTINGS = ("WXBOT_NTFY_TOPIC", "WXBOT_HEARTBEAT_TOPIC", "WXBOT_CONFIG", "KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY",
                 "KALSHI_PRIVATE_KEY_PATH", "KALSHI_BASE_URL")


@pytest.fixture(autouse=True)
def without_real_settings(monkeypatch):
    for k in REAL_SETTINGS:
        monkeypatch.setenv(k, "")       # recorded first, so whatever a test (or .env) sets is removed afterwards
        monkeypatch.delenv(k)
    yield


# Public copy: these tests need files kept out of this repository (the live and demo configs, the trained models and
# fitted pool weights, or the deploy scripts for the server). They run in the private repo; here they are skipped.
PRIVATE_FILE_TESTS = {
    "tests/test_a_subcap.py::test_no_sub_cap_by_default",
    "tests/test_adversary.py::test_a_fill_the_bot_did_not_book_fails_the_round",
    "tests/test_adversary.py::test_a_round_never_takes_more_than_its_limits",
    "tests/test_adversary.py::test_a_take_is_immediate_or_cancel_at_the_bots_price_on_the_other_side",
    "tests/test_adversary.py::test_chase_takes_a_new_order_as_soon_as_it_appears",
    "tests/test_adversary.py::test_rapid_takes_three_times_and_dust_takes_a_fraction",
    "tests/test_adversary.py::test_rests_never_cross_the_book",
    "tests/test_adversary.py::test_takes_only_hit_bot_orders_that_are_the_best_price_on_their_side",
    "tests/test_adversary.py::test_the_checks_pass_when_the_bot_booked_everything",
    "tests/test_adversary.py::test_vm_trouble_needs_account_2s_key_and_stop_leaves_a_flag",
    "tests/test_adversary.py::test_vm_trouble_start_installs_the_service_for_this_folder_and_clears_the_stop_flag",
    "tests/test_adversary.py::test_wall_and_undercut_rest_one_cent_off_the_bot_and_are_cancelled",
    "tests/test_demo_test.py::test_a_later_failure_removes_an_earlier_pass",
    "tests/test_demo_test.py::test_clock_time_is_not_running_time",
    "tests/test_demo_test.py::test_demo_start_begins_a_fresh_test_with_the_demo_key",
    "tests/test_demo_test.py::test_demo_start_does_not_start_a_second_demo_bot",
    "tests/test_demo_test.py::test_demo_start_stops_before_the_order_test_when_the_account_is_not_ready",
    "tests/test_demo_test.py::test_demo_stop_stops_the_service_and_cancels_with_the_demo_key",
    "tests/test_demo_test.py::test_go_live_stays_locked_until_the_demo_test_passed_on_the_deployed_code",
    "tests/test_demo_test.py::test_no_demo_fills_yet_is_a_check_not_a_pass",
    "tests/test_demo_test.py::test_one_night_passes_only_when_shourya_accepts_the_short_demo",
    "tests/test_demo_test.py::test_the_demo_gate_judges_the_demo_bot_only",
    "tests/test_demo_test.py::test_the_demo_service_plist_is_valid_and_runs_the_demo_wrapper",
    "tests/test_demo_test.py::test_the_demo_service_refuses_the_real_key",
    "tests/test_demo_test.py::test_the_scripts_parse",
    "tests/test_demo_test.py::test_the_short_demo_still_needs_a_clean_night",
    "tests/test_demo_test.py::test_too_little_activity_fails_and_leaves_go_live_locked",
    "tests/test_demo_test.py::test_two_clean_demo_days_pass_against_the_demo_exchange_and_unlock_go_live",
    "tests/test_demo_test.py::test_two_days_on_other_code_do_not_count",
    "tests/test_demo_test.py::test_two_full_days_pass_without_the_short_mark",
    "tests/test_demo_test.py::test_vm_demo_does_not_start_twice_and_stop_cancels_with_the_demo_key",
    "tests/test_demo_test.py::test_vm_demo_restart_keeps_the_record_and_starts_only_after_the_order_test",
    "tests/test_demo_test.py::test_vm_demo_start_runs_a_fresh_test_as_a_service_with_the_demo_key",
    "tests/test_demo_test.py::test_vm_demo_start_stops_before_the_order_test_when_the_account_is_not_ready",
    "tests/test_early_shadow.py::test_a_refit_outside_the_sanity_bounds_leaves_the_weights_alone",
    "tests/test_early_shadow.py::test_a_round5_model_gets_the_median_anchor_and_the_hourly_nbm",
    "tests/test_early_shadow.py::test_a_run_logs_the_frozen_rules_signals_once_per_bracket_and_side_over_both_hours",
    "tests/test_early_shadow.py::test_c_without_its_aifs_inputs_still_logs_and_says_so",
    "tests/test_early_shadow.py::test_candidates_b_and_c_log_to_their_own_databases_and_c_gets_aifs_inputs",
    "tests/test_early_shadow.py::test_g_logs_with_the_tilted_pool_and_stores_nbm",
    "tests/test_early_shadow.py::test_g_refit_keeps_the_tilt",
    "tests/test_early_shadow.py::test_monday_refit_replaces_only_the_early_hours_once_and_uses_the_shadows_own_snapshots",
    "tests/test_early_shadow.py::test_outside_the_slots_it_only_settles_and_a_dry_run_writes_nothing",
    "tests/test_early_shadow.py::test_refit_of_the_early_hours_reproduces_the_shipped_weights_on_the_same_rows",
    "tests/test_early_shadow.py::test_signals_book_the_depth_they_saw_and_size_keeps_what_sleeve_e_sends",
    "tests/test_early_shadow.py::test_status_gives_counts_only",
    "tests/test_early_shadow.py::test_the_execution_log_follows_each_signal_for_fifteen_minutes",
    "tests/test_early_shadow.py::test_the_execution_log_is_off_by_default_shared_by_candidates_and_stops_at_the_deadline",
    "tests/test_exec_log.py::test_the_option_is_off_by_default_and_on_only_where_the_configs_say",
    "tests/test_ml.py::test_numpy_booster_matches_lightgbm_and_loads_production_model",
    "tests/test_ml.py::test_production_artifacts_present_and_consistent",
    "tests/test_ml.py::test_tier1_parity_check_passes_on_both_engines_and_catches_a_wrong_engine",
    "tests/test_ml.py::test_tier1live_falls_back_to_numpy_without_lightgbm",
    "tests/test_rain.py::test_frozen_model_and_numpy_engine_agree",
    "tests/test_same_weather.py::test_the_configs",
    "tests/test_shutoff.py::test_the_statistic_matches_the_calibration_code",
    "tests/test_stop_guard.py::test_the_configs",
}


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="needs files kept private (live configs, trained models or deploy scripts)")
    for item in items:
        if item.nodeid.split("[")[0] in PRIVATE_FILE_TESTS:
            item.add_marker(skip)
