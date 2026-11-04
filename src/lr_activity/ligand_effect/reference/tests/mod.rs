mod build;
mod download;
mod programmes;
mod sources;
mod table;

/// An OmniPath `/interactions` TSV: rows of (source id, target id, source
/// symbol, target symbol, sign, supporting resources), directed by
/// consensus; a symbol with `_` is a complex.
fn ix_tsv(rows: &[(&str, &str, &str, &str, i8, &str)]) -> String {
    let mut s = String::from(
        "source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\t\
         is_inhibition\tconsensus_direction\tconsensus_stimulation\tconsensus_inhibition\t\
         sources\ttype\tentity_type_source\tentity_type_target\n",
    );
    let py = |b: bool| if b { "True" } else { "False" };
    let entity = |g: &str| {
        if g.contains('_') {
            "complex"
        } else {
            "protein"
        }
    };
    for &(a, b, ga, gb, sign, srcs) in rows {
        let (stim, inh) = (py(sign > 0), py(sign < 0));
        s.push_str(&format!(
            "{a}\t{b}\t{ga}\t{gb}\tTrue\t{stim}\t{inh}\tTrue\t{stim}\t{inh}\t{srcs}\t\
             post_translational\t{}\t{}\n",
            entity(ga),
            entity(gb)
        ));
    }
    s
}

/// A `/resources` licence table: `Free` and `Foo_CollecTRI` redistributable,
/// `NC` non-commercial, `SA` share-alike, `HPRD` academic.
fn resources_json() -> &'static str {
    r#"{
      "Free": {"license": {"name": "CC BY 4.0", "purpose": "commercial", "sharing": "free"}},
      "Foo_CollecTRI": {"license": {"name": "CC BY 4.0", "purpose": "for_profit", "sharing": "free"}},
      "NC": {"license": {"name": "CC BY-NC 4.0", "purpose": "academic", "sharing": "free"}},
      "SA": {"license": {"name": "CC BY-SA 4.0", "purpose": "commercial", "sharing": "alike"}},
      "HPRD": {"license": {"name": "custom", "purpose": "academic", "sharing": "deriv"}},
      "Bare": {}
    }"#
}
