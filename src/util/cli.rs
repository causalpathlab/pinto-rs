//! How the command line names the values of a [`ValueEnum`].

use clap::ValueEnum;

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
