use super::*;

#[test]
fn an_error_popup_wraps_its_message_to_fit() {
    let msg = format!(
        "lupin annotate: {} is unreadable\nsecond line",
        "x/".repeat(60)
    );
    let lines: Vec<String> = error_lines(&msg, 40)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert!(lines.iter().all(|l| l.chars().count() <= 38), "{lines:?}");
    let text: String = lines.iter().map(|l| l.trim_start()).collect();
    assert!(text.contains(&"x/".repeat(60)), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("second line")));
    assert!(lines.last().unwrap().contains("closes"));
}
