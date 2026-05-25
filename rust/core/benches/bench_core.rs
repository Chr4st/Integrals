use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neurips_core::diff::differentiate;
use neurips_core::expr::{to_prefix_string, ExprNode};
use neurips_core::gen::{generate_batch, generate_random_tree, GenConfig, GenMode};

fn build_nested_expr(depth: usize) -> ExprNode {
    if depth == 0 {
        return ExprNode::var("x");
    }
    let inner = build_nested_expr(depth - 1);
    match depth % 3 {
        0 => ExprNode::sin(inner),
        1 => ExprNode::mul(inner, ExprNode::var("x")),
        _ => ExprNode::exp(inner),
    }
}

fn bench_differentiate(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiate");

    for depth in [5, 10, 15] {
        let expr = build_nested_expr(depth);
        group.bench_function(format!("depth_{depth}"), |b| {
            b.iter(|| differentiate(black_box(&expr), "x"))
        });
    }
    group.finish();
}

fn bench_to_prefix_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_prefix_string");

    for depth in [5, 10, 15] {
        let expr = build_nested_expr(depth);
        group.bench_function(format!("depth_{depth}"), |b| {
            b.iter(|| to_prefix_string(black_box(&expr)))
        });
    }
    group.finish();
}

fn bench_generate_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_batch");
    let config = GenConfig::default();

    for n in [100, 1_000, 10_000] {
        group.bench_function(format!("n_{n}"), |b| {
            b.iter(|| generate_batch(black_box(n), GenMode::Univariate, &config))
        });
    }
    group.finish();
}

fn bench_generate_random_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_random_tree");

    for max_depth in [4, 6, 10] {
        let config = GenConfig {
            max_depth,
            ..GenConfig::default()
        };
        group.bench_function(format!("depth_{max_depth}"), |b| {
            let mut rng = rand::thread_rng();
            b.iter(|| generate_random_tree(black_box(&config), &mut rng))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_differentiate,
    bench_to_prefix_string,
    bench_generate_batch,
    bench_generate_random_tree,
);
criterion_main!(benches);
