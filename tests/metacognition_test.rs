//! End-to-end tests for the MERIT metacognition engine.
//!
//! These exercise the engine via its public façade (no internals) so they
//! double as documentation for the intended agent-loop integration.

use agentrust::metacognition::{
    EngineConfig, FailureEvent, MetaAction, MetacognitionEngine, ProposalKind, ReviserConfig,
    SchemaProposal, SchemaReviser, SelfBelief, SelfModelStore, SelfProposition,
};

#[test]
fn full_turn_lifecycle_never_picks_act_under_repeated_errors() {
    let mut eng = MetacognitionEngine::new();
    eng.on_turn_start("refactor the authentication module");

    // Iteration 0: nothing observed yet. Anything but Abort is allowed.
    let d = eng.before_next_iteration(0);
    assert_ne!(d.action, MetaAction::Abort);

    // Two tool errors in a row — Reflect should win on iter 1.
    eng.on_tool_use("file_edit", r#"{"path":"/auth/mod.rs"}"#);
    eng.on_tool_result("file_edit", true);
    eng.on_tool_use("file_edit", r#"{"path":"/auth/mod.rs"}"#);
    eng.on_tool_result("file_edit", true);
    let d = eng.before_next_iteration(1);
    assert_eq!(d.action, MetaAction::Reflect);
    assert!(!d.injection.is_empty());

    // Now grind through more failures + lack of progress. Under sustained
    // negative signal the controller must not return to Act.
    for i in 2..6 {
        eng.on_tool_use("file_edit", &format!(r#"{{"i":{}}}"#, i));
        eng.on_tool_result("file_edit", true);
        let d = eng.before_next_iteration(i);
        assert_ne!(d.action, MetaAction::Act, "iter {}", i);
    }

    eng.on_turn_end();
}

#[test]
fn cot_intervention_overrides_act_when_loop_detected_after_priming() {
    // Seed lots of positive evidence so the EFE policy picks Act at iter 0
    // (high progress/reasoning/tool means make Act's utility dominate).
    let mut eng = MetacognitionEngine::new();
    for _ in 0..40 {
        eng.controller_mut().observe_tool_result("execute_command", false);
        eng.controller_mut().observe_cot_quality(0.95);
        eng.controller_mut().observe_progress(true);
    }
    eng.on_turn_start("anything");
    // Two identical tool calls → loop_detected = true.
    eng.on_tool_use("file_read", "path=/a");
    eng.on_tool_use("file_read", "path=/a");
    let d = eng.before_next_iteration(0);
    assert_eq!(d.action, MetaAction::Act, "primed controller should pick Act");
    assert!(!d.injection.is_empty(), "loop_detected should override");
    assert!(d.reason.contains("cot_intervention"));
}

#[test]
fn layer3_prompt_section_grows_after_seeding_propositions() {
    let mut eng = MetacognitionEngine::new();
    eng.on_turn_start("write a rust async parser");
    assert!(eng.self_model_prompt_section().is_empty());

    eng.store_mut().add_proposition(SelfProposition::new(
        "I tend to forget to .await futures in Rust async code",
        vec!["rust".into(), "async".into()],
    ));
    let s = eng.self_model_prompt_section();
    assert!(s.contains("Self-knowledge"));
    assert!(s.contains("await"));
}

#[test]
fn layer4_promotes_repeated_failure_cluster_into_a_proposition() {
    let mut reviser = SchemaReviser::with_config(ReviserConfig {
        min_evidence_count: 3,
        review_every_n_episodes: 1,
        cluster_jaccard_thresh: 0.3,
        novelty_threshold: 0.5,
        max_failures_buffer: 200,
    });
    // Five very similar failures.
    for _ in 0..5 {
        reviser.record_failure(FailureEvent {
            ts: 0,
            tool: "file_edit".into(),
            description: "file_edit failed: path not found in workspace".into(),
            task_type: String::new(),
        });
    }
    reviser.note_episode_complete();
    assert!(reviser.should_review());

    let belief = SelfBelief::new();
    let store = SelfModelStore::new();
    let props = reviser.propose_revisions(&belief, &store);
    assert!(!props.is_empty(), "should propose at least one revision");
    let p: &SchemaProposal = props.first().unwrap();
    assert!(p.evidence_count >= 3);
    assert!(p.novelty > 0.0);

    // Apply it and verify the store / belief changed accordingly.
    let mut belief = belief;
    let mut store = store;
    let applied = reviser.apply(p, &mut belief, &mut store);
    assert!(applied);
    match p.kind {
        ProposalKind::AddProposition => assert_eq!(store.len(), 1),
        ProposalKind::AddDimension => assert!(belief.has_dimension(&p.name_or_text)),
    }
}

#[test]
fn engine_respects_disable_flags() {
    let cfg = EngineConfig {
        enable_layer3: false,
        enable_layer4: false,
        ..Default::default()
    };
    let mut eng = MetacognitionEngine::with_config(cfg);
    eng.on_turn_start("hello");
    eng.store_mut().add_proposition(SelfProposition::new(
        "I am terse with hello-world tasks",
        vec!["greeting".into()],
    ));
    // Layer 3 disabled → no section even though the store is populated.
    assert!(eng.self_model_prompt_section().is_empty());
}

#[test]
fn self_belief_overall_competence_rises_with_successes() {
    let mut eng = MetacognitionEngine::new();
    let before = eng.controller().belief().overall_competence();
    for _ in 0..50 {
        eng.on_tool_use("execute_command", "ls");
        eng.on_tool_result("execute_command", false);
    }
    let after = eng.controller().belief().overall_competence();
    assert!(
        after > before,
        "overall competence should rise: before={before}, after={after}"
    );
}
