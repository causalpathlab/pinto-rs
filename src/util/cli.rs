//! How the command line names the values of a [`ValueEnum`], and which
//! flags it was given.

use clap::ValueEnum;

/// The arguments `ids` of the (sub)command `cmd` that its matches `m` took
/// from the command line, not from a default, each as its `--long` flag.
pub fn given_on_cli<'a>(
    cmd: &clap::Command,
    m: &clap::ArgMatches,
    ids: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    ids.into_iter()
        .filter(|id| m.value_source(id) == Some(clap::parser::ValueSource::CommandLine))
        .filter_map(|id| cmd.get_arguments().find(|a| a.get_id() == id))
        .filter_map(|a| a.get_long().map(|l| format!("--{l}")))
        .collect()
}

/// The argument ids of the group `group` of `cmd` (a derived `Args`
/// struct's group is named after the struct), empty when it has none.
pub fn group_ids<'c>(cmd: &'c clap::Command, group: &str) -> Vec<&'c str> {
    cmd.get_groups()
        .filter(|g| g.get_id() == group)
        .flat_map(|g| g.get_args())
        .map(clap::Id::as_str)
        .collect()
}

/// How the command line spells `v`. pinto's value enums skip no variant.
pub fn value_name<T: ValueEnum>(v: &T) -> String {
    v.to_possible_value()
        .expect("no skipped variants")
        .get_name()
        .to_string()
}

/// The values a flag of type `T` takes, as the command line names them.
#[cfg(feature = "view")]
pub fn choices<T: ValueEnum>() -> Vec<String> {
    T::value_variants()
        .iter()
        .filter_map(|v| v.to_possible_value())
        .map(|p| p.get_name().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(clap::ValueEnum, Clone)]
    enum Pick {
        One,
        NoRefit,
    }

    #[test]
    fn values_are_named_as_the_command_line_spells_them() {
        assert_eq!(value_name(&Pick::One), "one");
        assert_eq!(value_name(&Pick::NoRefit), "no-refit");
        #[cfg(feature = "view")]
        assert_eq!(choices::<Pick>(), ["one", "no-refit"]);
    }
}
