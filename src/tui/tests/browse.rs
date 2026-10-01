use super::*;

/// Text files, several at once when `many`; `.zarr` stores too.
struct Txt {
    many: bool,
}

/// Text files, refusing any named `no.txt`.
struct Picky;

impl Wanted for Picky {
    type About = ();

    fn header(&self) -> Header {
        Txt { many: false }.header()
    }

    fn file(&self, _path: &Path, name: &str) -> Option<()> {
        name.ends_with(".txt").then_some(())
    }

    fn describe<'a>(&self, (): &'a ()) -> std::borrow::Cow<'a, str> {
        "".into()
    }

    fn refuse(&self, name: &str, (): &()) -> Option<String> {
        (name == "no.txt").then(|| format!("{name} will not do"))
    }
}

impl Wanted for Txt {
    type About = String;

    fn header(&self) -> Header {
        Header {
            title: "Text".into(),
            notes: Vec::new(),
            what: "text files",
            star: None,
            verb: "take",
            columns: None,
        }
    }

    fn file(&self, path: &Path, name: &str) -> Option<String> {
        name.ends_with(".txt").then(|| size_of(path))
    }

    fn store(&self, _path: &Path, _name: &str) -> Option<String> {
        Some(String::new())
    }

    fn describe<'a>(&self, about: &'a String) -> std::borrow::Cow<'a, str> {
        std::borrow::Cow::Borrowed(about)
    }

    fn many(&self) -> bool {
        self.many
    }
}

fn write(dir: &Path, name: &str) {
    std::fs::write(dir.join(name), "").unwrap();
}

fn key(c: KeyCode) -> KeyEvent {
    KeyEvent::new(c, KeyModifiers::NONE)
}

/// The files a key took; none when it took none.
fn taken(o: Outcome) -> Vec<PathBuf> {
    match o {
        Outcome::Chosen(c) => c.files(),
        _ => Vec::new(),
    }
}

fn names<W: Wanted>(b: &Browser<W>) -> Vec<String> {
    b.shown().iter().map(|e| e.name().to_string()).collect()
}

#[test]
fn folders_come_before_wanted_files_and_stores_are_files() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(d, "b.txt");
    write(d, "a.txt");
    write(d, "skip.csv");
    std::fs::create_dir(d.join("sub")).unwrap();
    std::fs::create_dir(d.join("s.zarr")).unwrap();
    std::fs::create_dir(d.join(".hidden")).unwrap();
    let mut b = Browser::open(d.to_path_buf(), Txt { many: false }, None);
    assert_eq!(names(&b), ["..", "sub", "a.txt", "b.txt", "s.zarr"]);
    // Hidden entries show only when asked for with a leading `.`.
    b.filter.push('.');
    assert!(names(&b).contains(&".hidden".to_string()));
}

#[test]
fn enter_chooses_one_file_and_opens_folders() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    write(&dir.path().join("sub"), "a.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Txt { many: false }, None);
    assert_eq!(b.current().map(Entry::name), Some("sub"));
    assert_eq!(b.key(key(KeyCode::Enter)), Outcome::Moved);
    assert_eq!(b.current().map(Entry::name), Some("a.txt"));
    // Space narrows a single-choice browser like any letter.
    assert_eq!(b.key(key(KeyCode::Char(' '))), Outcome::Moved);
    assert_eq!(b.filter, " ");
    b.key(key(KeyCode::Backspace));
    assert_eq!(
        taken(b.key(key(KeyCode::Enter))),
        [dir.path().join("sub/a.txt")]
    );
    b.go_up();
    assert_eq!(b.current().map(Entry::name), Some("sub"));
}

#[test]
fn marked_files_in_several_folders_are_taken_together() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.txt");
    std::fs::create_dir(dir.path().join("more")).unwrap();
    write(&dir.path().join("more"), "b.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Txt { many: true }, None);
    assert_eq!(names(&b), ["..", "more", "a.txt"]);
    b.row = 2;
    b.key(key(KeyCode::Char(' ')));
    b.row = 1;
    assert_eq!(
        b.key(key(KeyCode::Enter)),
        Outcome::Moved,
        "a folder still opens"
    );
    assert_eq!(b.dir, dir.path().join("more"));
    b.row = 1;
    b.key(key(KeyCode::Char(' ')));
    b.row = 1;
    assert_eq!(
        taken(b.key(key(KeyCode::Enter))),
        [dir.path().join("a.txt"), dir.path().join("more/b.txt")]
    );
    assert!(b.marked.is_empty());
}

#[test]
fn sizes_read_in_their_unit() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("f");
    std::fs::write(&p, vec![0u8; 1500]).unwrap();
    assert_eq!(size_of(&p), "1.5 kB");
    std::fs::write(&p, b"abc").unwrap();
    assert_eq!(size_of(&p), "3 B");
}

#[test]
fn a_refused_file_is_not_taken_and_the_popup_says_why() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "no.txt");
    write(dir.path(), "yes.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Picky, Some("no.txt"));
    assert_eq!(b.key(key(KeyCode::Enter)), Outcome::Moved);
    let said: Vec<String> = b.lines(20, 80).iter().map(ToString::to_string).collect();
    assert!(
        said.iter().any(|l| l.contains("no.txt will not do")),
        "{said:?}"
    );
    b.key(key(KeyCode::Down));
    assert!(b.refused.is_none());
    assert_eq!(
        taken(b.key(key(KeyCode::Enter))),
        [dir.path().join("yes.txt")]
    );
}

#[test]
fn lines_fit_their_width_and_a_caller_lays_out_its_rows() {
    struct Counted;
    impl Wanted for Counted {
        type About = usize;
        fn header(&self) -> Header {
            Header {
                title: "a title much longer than the panel it is drawn in".into(),
                notes: Vec::new(),
                what: "files",
                star: Some("the best one"),
                verb: "take",
                columns: Some("     n  name"),
            }
        }
        fn file(&self, _path: &Path, name: &str) -> Option<usize> {
            name.ends_with(".txt").then_some(name.len())
        }
        fn describe<'a>(&self, n: &'a usize) -> std::borrow::Cow<'a, str> {
            n.to_string().into()
        }
        fn row(&self, name: &str, n: &usize, _name_w: usize) -> String {
            format!("{n:>5}  {name}")
        }
        fn best(&self, _dir: &Path, files: &[(&str, &usize)]) -> Option<String> {
            files
                .iter()
                .max_by_key(|(_, n)| **n)
                .map(|(f, _)| f.to_string())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.txt");
    write(dir.path(), "longer.txt");
    let b = Browser::open(dir.path().to_path_buf(), Counted, None);
    assert_eq!(b.current().map(Entry::name), Some("longer.txt"));
    let said: Vec<String> = b.lines(20, 30).iter().map(ToString::to_string).collect();
    assert!(said.iter().all(|l| l.chars().count() <= 30), "{said:?}");
    assert!(said.contains(&"*   10  longer.txt".to_string()), "{said:?}");
    assert!(said.contains(&"     n  name".to_string()), "{said:?}");
}

/// Text files that can be taken several at once, refusing `no.txt`.
struct PickyMany;

impl Wanted for PickyMany {
    type About = ();

    fn header(&self) -> Header {
        Picky.header()
    }

    fn file(&self, path: &Path, name: &str) -> Option<()> {
        Picky.file(path, name)
    }

    fn describe<'a>(&self, (): &'a ()) -> std::borrow::Cow<'a, str> {
        "".into()
    }

    fn refuse(&self, name: &str, (): &()) -> Option<String> {
        Picky.refuse(name, &())
    }

    fn many(&self) -> bool {
        true
    }
}

#[test]
fn refused_files_are_never_marked() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "no.txt");
    write(dir.path(), "yes.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), PickyMany, Some("no.txt"));
    b.key(key(KeyCode::Char(' ')));
    assert!(b.marked.is_empty());
    assert_eq!(b.refused.as_deref(), Some("no.txt will not do"));
    b.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
    assert_eq!(
        b.marked.iter().collect::<Vec<_>>(),
        [&dir.path().join("yes.txt")]
    );
    assert!(b.refused.is_some(), "the refused one is named");
}

#[test]
fn a_refusal_stays_through_keys_that_do_nothing() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "no.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Picky, Some("no.txt"));
    b.key(key(KeyCode::Enter));
    assert!(b.refused.is_some());
    assert_eq!(b.key(key(KeyCode::Tab)), Outcome::Ignored);
    assert!(b.refused.is_some());
}

#[test]
fn the_key_hints_fit_however_short_the_panel() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..40 {
        write(dir.path(), &format!("f{i:02}.txt"));
    }
    let b = Browser::open(dir.path().to_path_buf(), Txt { many: true }, None);
    let lines = b.lines(20, 80);
    assert!(lines.len() <= 20, "{}", lines.len());
    assert!(lines.last().unwrap().to_string().contains("Esc cancel"));
}

/// Counts the files it is asked about.
struct Counting(std::cell::Cell<usize>);

impl Wanted for Counting {
    type About = ();

    fn header(&self) -> Header {
        Txt { many: false }.header()
    }

    fn file(&self, _path: &Path, name: &str) -> Option<()> {
        self.0.set(self.0.get() + 1);
        name.ends_with(".txt").then_some(())
    }

    fn describe<'a>(&self, (): &'a ()) -> std::borrow::Cow<'a, str> {
        "".into()
    }

    fn best(&self, _dir: &Path, files: &[(&str, &())]) -> Option<String> {
        files.first().map(|(n, _)| n.to_string())
    }
}

#[test]
fn hidden_files_are_read_only_when_asked_for_and_never_start_the_cursor() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), ".a.txt");
    write(dir.path(), "b.txt");
    let mut b = Browser::open(dir.path().to_path_buf(), Counting(0.into()), None);
    assert_eq!(b.want.0.get(), 1, "the dotfile is not opened");
    assert_eq!(b.current().map(Entry::name), Some("b.txt"));
    b.key(key(KeyCode::Char('.')));
    b.key(key(KeyCode::Char('a')));
    assert_eq!(names(&b), ["..", ".a.txt"]);
    b.key(key(KeyCode::Backspace));
    b.key(key(KeyCode::Backspace));
    assert_eq!(names(&b), ["..", "b.txt"]);
}
