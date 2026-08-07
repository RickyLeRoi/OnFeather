import pytest

from onfeather_solo.memory import STATUS_CONFIRMED, STATUS_PROPOSED, create
from onfeather_solo.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def test_save_writes_a_markdown_file(store):
    path = store.save(create("a fact"))
    assert path.suffix == ".md"
    assert path.read_text().startswith("---")


def test_proposed_memories_land_in_the_proposed_directory(store):
    path = store.save(create("a fact"))
    assert path.parent.name == "proposed"


def test_confirming_moves_the_file(store):
    memory = create("a fact")
    old = store.save(memory)
    memory.confirm()
    new = store.save(memory)

    assert new.parent.name == "confirmed"
    assert not old.exists(), "the proposed copy must not linger"


def test_round_trip_through_disk(store):
    original = create("something worth keeping", tags=["x"], source="chat")
    store.save(original)

    loaded = store.get(original.id)
    assert loaded.body == original.body
    assert loaded.tags == ["x"]


def test_add_is_idempotent(store):
    first, new_first = store.add(create("the same fact"))
    second, new_second = store.add(create("the same fact"))

    assert new_first and not new_second
    assert first.id == second.id
    assert len(store.all()) == 1


def test_directory_wins_over_frontmatter(store):
    """A file moved by hand is a decision, not a mistake to correct."""
    memory = create("a fact")
    store.save(memory)

    confirmed_dir = store.root / "confirmed"
    confirmed_dir.mkdir(parents=True, exist_ok=True)
    moved = confirmed_dir / memory.path.name
    moved.write_text(memory.path.read_text())
    memory.path.unlink()

    assert store.get(memory.id).status == STATUS_CONFIRMED


def test_malformed_files_are_skipped(store):
    store.save(create("a good one"))
    bad = store.root / "proposed" / "broken.md"
    bad.write_text("no frontmatter here")

    assert len(store.all()) == 1


def test_get_accepts_an_unambiguous_prefix(store):
    memory = create("unique content here")
    store.save(memory)
    assert store.get(memory.id[:6]).id == memory.id


def test_get_returns_none_for_unknown_ids(store):
    assert store.get("nope") is None


def test_delete_removes_the_file(store):
    memory = create("temporary")
    store.save(memory)
    store.delete(memory)
    assert store.get(memory.id) is None


def test_empty_store_lists_nothing(store):
    assert store.all() == []


# -- counts ---------------------------------------------------------------


def test_counts_an_empty_store_as_zeros(store):
    assert store.counts() == {"proposed": 0, "confirmed": 0, "rejected": 0}


def test_counts_every_status(store):
    store.save(create("still to review"))
    store.save(create("also to review"))

    accepted = create("kept")
    accepted.confirm()
    store.save(accepted)

    discarded = create("thrown out")
    discarded.reject()
    store.save(discarded)

    assert store.counts() == {"proposed": 2, "confirmed": 1, "rejected": 1}


def test_counts_agree_with_by_status(store):
    for index in range(3):
        store.save(create(f"memory {index}"))
    assert store.counts()[STATUS_PROPOSED] == len(store.by_status(STATUS_PROPOSED))


def test_counts_reads_no_files(store, monkeypatch):
    """The whole point: a monitoring poll must not parse the collection."""
    store.save(create("a fact"))
    monkeypatch.setattr(
        Store, "_read", lambda *_: pytest.fail("counts() must not open a memory")
    )
    assert store.counts()["proposed"] == 1


def test_counts_ignores_files_that_are_not_memories(store):
    (store.root / "proposed").mkdir(parents=True)
    (store.root / "proposed" / "notes.txt").write_text("not a memory")
    assert store.counts()["proposed"] == 0


# -- search ---------------------------------------------------------------


def confirmed(store, text, **kwargs):
    memory = create(text, **kwargs)
    memory.confirm()
    store.save(memory)
    return memory


def test_search_finds_whole_words(store):
    confirmed(store, "Riccardo prefers Apache licences")
    confirmed(store, "the cat sat on the mat")

    hits = store.search("apache")
    assert len(hits) == 1
    assert "Apache" in hits[0].memory.body


def test_search_ranks_more_matches_higher(store):
    confirmed(store, "python testing with pytest")
    confirmed(store, "python only")

    hits = store.search("python testing")
    assert hits[0].memory.body.startswith("python testing")


def test_search_matches_tags(store):
    confirmed(store, "some content", tags=["licensing"])
    assert store.search("licensing")


def test_search_ignores_proposed_by_default(store):
    """Unreviewed memories must not leak into answers."""
    store.save(create("unreviewed but relevant apache"))
    assert store.search("apache") == []
    assert store.search("apache", status=None)


def test_search_weights_by_confidence(store):
    high = create("apache licence detail", confidence=1.0)
    low = create("apache licence note", confidence=0.2)
    for memory in (high, low):
        memory.confirm()
        store.save(memory)

    hits = store.search("apache licence")
    assert hits[0].memory.id == high.id


def test_search_respects_the_limit(store):
    for index in range(5):
        confirmed(store, f"shared term number {index}")
    assert len(store.search("shared", limit=2)) == 2


def test_search_with_no_usable_terms(store):
    confirmed(store, "content")
    assert store.search("  ") == []


def test_search_returns_nothing_for_misses(store):
    confirmed(store, "content")
    assert store.search("absent") == []


# -- stable filenames -----------------------------------------------------


def test_editing_does_not_rename_the_file(store):
    """A `[[link]]` names a file. If correcting a memory renamed it, every link
    pointing at it would break — and correcting memories is the whole product."""
    memory = create("Riccardo prefers Apache licences")
    first = store.save(memory)

    memory.edit("Riccardo prefers the MIT licence after all")
    assert store.save(memory).name == first.name
    assert first.exists()


def test_changing_status_keeps_the_filename(store):
    memory = create("a fact")
    first = store.save(memory)
    memory.confirm()

    moved = store.save(memory)
    assert moved.name == first.name
    assert moved.parent.name == "confirmed"


def test_a_hand_chosen_filename_survives(store):
    """Renaming a note in Obsidian updates the links pointing at it. Renaming it
    back on the next write would undo that silently."""
    memory = create("a fact")
    original = store.save(memory)
    renamed = original.with_name("my-own-name.md")
    original.rename(renamed)

    reloaded = store.get(memory.id)
    reloaded.edit("a corrected fact")
    assert store.save(reloaded).name == "my-own-name.md"


# -- links ----------------------------------------------------------------


def test_links_resolve_by_filename(store):
    target = confirmed(store, "the licence decision")
    source = confirmed(store, f"a fact, see [[{target.path.stem}]]")

    assert store.resolve(target.path.stem).id == target.id
    assert store.links(source)[0][1].id == target.id


def test_links_resolve_by_id_and_by_prefix(store):
    target = confirmed(store, "the licence decision")
    assert store.resolve(target.id).id == target.id
    assert store.resolve(target.id[:6]).id == target.id


def test_link_resolution_ignores_case_in_filenames(store):
    target = confirmed(store, "the licence decision")
    assert store.resolve(target.path.stem.upper()).id == target.id


def test_an_unresolved_link_is_reported_not_dropped(store):
    """A dangling link marks a memory worth writing."""
    source = confirmed(store, "a fact, see [[nothing-here]]")
    assert store.links(source) == [("nothing-here", None)]


def test_backlinks_find_what_points_here(store):
    target = confirmed(store, "the licence decision")
    source = confirmed(store, f"a fact, see [[{target.path.stem}]]")

    assert [m.id for m in store.backlinks(target)] == [source.id]
    assert store.backlinks(source) == []


def test_a_memory_does_not_link_to_itself(store):
    memory = confirmed(store, "a fact")
    memory.edit(f"a fact, see [[{memory.path.stem}]]")
    store.save(memory)

    assert store.backlinks(store.get(memory.id)) == []


# -- search along links ---------------------------------------------------


def test_search_reaches_a_neighbour_that_does_not_match(store):
    """The point of linking: a memory with none of the query's words comes back
    because something that did match points at it."""
    neighbour = confirmed(store, "the whole team moved to Berlin")
    confirmed(store, f"Riccardo prefers Apache licences [[{neighbour.path.stem}]]")

    found = {hit.memory.id: hit for hit in store.search("apache")}
    assert neighbour.id in found
    assert found[neighbour.id].via is not None


def test_a_backlink_is_followed_too(store):
    """Being linked *from* a hit is as good a lead as linking to one."""
    neighbour = confirmed(store, "the whole team moved to Berlin")
    match = confirmed(store, "Riccardo prefers Apache licences")
    neighbour.edit(f"the whole team moved to Berlin [[{match.path.stem}]]")
    store.save(neighbour)

    assert neighbour.id in {hit.memory.id for hit in store.search("apache")}


def test_a_neighbour_never_outranks_what_pulled_it_in(store):
    neighbour = confirmed(store, "the whole team moved to Berlin")
    match = confirmed(store, f"Riccardo prefers Apache licences [[{neighbour.path.stem}]]")

    hits = store.search("apache")
    assert hits[0].memory.id == match.id
    assert hits[0].via is None


def test_a_direct_match_is_never_reported_as_a_link_result(store):
    """It matched on its own; saying otherwise would misexplain the result."""
    weak = confirmed(store, "apache")
    confirmed(store, f"Riccardo prefers Apache licences and apache things [[{weak.path.stem}]]")

    by_id = {hit.memory.id: hit for hit in store.search("apache licences")}
    assert by_id[weak.id].via is None


def test_links_do_not_leak_unreviewed_memories_into_search(store):
    """A proposal must not reach an answer by being linked from a confirmed one."""
    unreviewed = create("unreviewed but linked")
    store.save(unreviewed)
    confirmed(store, f"Riccardo prefers Apache licences [[{unreviewed.path.stem}]]")

    assert unreviewed.id not in {hit.memory.id for hit in store.search("apache")}


def test_traversal_stops_after_one_hop(store):
    far = confirmed(store, "something entirely unrelated")
    near = confirmed(store, f"the whole team moved to Berlin [[{far.path.stem}]]")
    confirmed(store, f"Riccardo prefers Apache licences [[{near.path.stem}]]")

    found = {hit.memory.id for hit in store.search("apache")}
    assert near.id in found
    assert far.id not in found


def test_following_links_can_be_turned_off(store):
    neighbour = confirmed(store, "the whole team moved to Berlin")
    confirmed(store, f"Riccardo prefers Apache licences [[{neighbour.path.stem}]]")

    hits = store.search("apache", follow_links=False)
    assert [hit.memory.id for hit in hits] != []
    assert neighbour.id not in {hit.memory.id for hit in hits}


def test_the_limit_still_holds_with_links(store):
    for index in range(5):
        neighbour = confirmed(store, f"unrelated neighbour {index}")
        confirmed(store, f"shared term {index} [[{neighbour.path.stem}]]")

    assert len(store.search("shared", limit=3)) == 3
