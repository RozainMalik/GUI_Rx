"""Every system and user prompt as a constant, plus prompt_hash() so a record
carries proof of which prompts produced it. Templates use {name} placeholders
and are filled with render(), which leaves the JSON skeletons' braces alone."""
from __future__ import annotations

import hashlib
import sys

ELEMENT_KINDS = ("toolbar button", "menu item", "list row", "table row", "tab", "checkbox",
                 "radio button", "text field", "dropdown", "slider", "icon button", "link", "other")
STATES = ("pressed", "highlighted", "selected", "hovered", "focused", "plain")
REGION_KINDS = ("menu", "dialog", "toolbar", "panel", "tab strip", "list", "other")

SYSTEM_PLAIN = "You are a helpful assistant."

SYSTEM_ANNOTATOR = """You are an expert annotator building a click-grounding training dataset from
screenshots of medical-imaging desktop applications (DICOM viewers,
segmentation tools, image-analysis suites). A student model will be trained on
your output: an instruction a user might give, paired with the bounding box of
the single control that carries it out.

Rules that are never broken:
1. Only what is visible. Never name a label, icon or state the image does not
   show. If you are unsure whether a control carries an icon, it does not.
2. The target is ONE interactive control: a button, menu item, list row, tab,
   checkbox, radio button, text field, slider or icon. Never a panel, a group,
   a whole toolbar, or a text label sitting beside a control.
3. Labels are quoted verbatim, character for character, including "..." and
   capitalisation.
4. Never mention the mouse cursor, hovering, highlighting, focus rings, this
   annotation task, or the screenshot itself.
5. Coordinates are [x1, y1, x2, y2] - left, top, right, bottom - as integers
   {coords} the width and height of THE IMAGE YOU ARE LOOKING AT.
   Read the control's edges off the image; do not round to convenient numbers.
6. Think as long as you need, then output exactly one JSON object and nothing
   after it."""

PROPOSE_USER = """Application: {app}

This is one screenshot of the application. Work out what the user is doing on
this screen, and where.

Step 1 - Intent. In one sentence, what is the user in the middle of doing? An
open menu, a dialog in front, a pressed or highlighted control, a selected row,
an expanded section, or a tool panel with fresh input all tell you where the
work is.

Step 2 - Focus region. Box the one area where that work is happening: the open
menu, the dialog, the toolbar strip, the panel. It is a region of the screen,
never the whole screen. If nothing is being worked on, the focus region is the
most prominent interactive area.

Step 3 - Candidates. Inside the focus region, list up to three interactive
controls the user would act on next, best first. Rank 1 is the control most
clearly being acted on - pressed, highlighted, selected, hovered, or the
obvious next step of the intent. Each candidate is a single control with its
own box.

If no interactive control on this screen can be identified with confidence,
return an empty candidates list. That is a correct answer; inventing a control
is not.

Return exactly one JSON object:
{"intent": "...",
 "focus_region": {"bbox": [x1, y1, x2, y2],
                  "kind": "menu|dialog|toolbar|panel|tab strip|list|other",
                  "why": "..."},
 "candidates": [
   {"rank": 1, "element_name": "...", "element_kind": "...",
    "text_on_screen": "...", "icon": "", "state": "...",
    "bbox": [x1, y1, x2, y2]}
 ]}

element_kind is exactly one of: {kinds}
state is exactly one of: pressed, highlighted, selected, hovered, focused, plain
text_on_screen is the control's own visible label, verbatim, or "" if none.
icon is a short description of the icon drawn on the control ("floppy disk",
"magnifier"), or "" if the control has no icon."""

DESCRIBE_USER = """Application: {app}

This image is a ZOOMED CROP of one region of a screenshot - not the whole
screen. It shows the {region_kind} the user is working in.

The control to annotate is:
  element_name:   {element_name}
  text_on_screen: {text_on_screen}
  icon:           {icon}
This control is fixed. Correct its label or icon if the zoom shows them
differently, but do not switch to a different control. If this control is
genuinely not in the image, return {"element_present": false} and nothing else.

Do the following, in order:
1. Find the control. Read its label again at this magnification and set
   text_on_screen and icon to exactly what is there.
2. Box it tightly: the whole clickable control - the full menu row, the whole
   button, a checkbox together with its own label - but not its neighbours and
   not the region around it.
3. Record its neighbours: the labels of the controls immediately above, below,
   left and right of it ("" where there is none), and its position ("8th of 24
   rows in the File menu", "3rd button from the left in the toolbar").
4. Write three instructions. Each is a COMMAND to a user, starts with a verb,
   and names this one control:
   - functional: what the action accomplishes.
       "Start a slideshow of the loaded images."
   - visual: what the control looks like - its verbatim label, and its icon
     only if it has one.
       "Click the menu item labelled 'Slideshow...'."
   - spatial: where it is - an ordinal, a neighbour, or a region.
       "Click the eighth row of the open File menu, just below 'Thumbnails'."

Never begin an instruction with A, An, The, This, Located, Opens, Displays,
Shows, Selects or Clicking. Never mention hovering, highlighting, the cursor,
or a bounding box.

Return exactly one JSON object:
{"element_present": true,
 "bbox": [x1, y1, x2, y2],
 "text_on_screen": "...", "icon": "...",
 "neighbours": {"above": "...", "below": "...", "left": "...", "right": "..."},
 "position": "...",
 "instructions": {"functional": "...", "visual": "...", "spatial": "..."}}"""

GROUND_USER = """Locate the element in this screenshot that the following instruction refers
to: "{instruction}"
Output only its bounding box as JSON:
[{"bbox_2d": [x1, y1, x2, y2], "label": "<short name>"}]
Coordinates are integers {coords} the image width and height."""

VERIFY_USER = """Image 1 is a screenshot with a red rectangle drawn on it. Image 2 is a
magnified view of the area around that rectangle, with the same rectangle
drawn.

The red rectangle is a PROPOSED answer for this instruction: "{visual}"
(also described as: "{functional}"). It may be right or wrong.

Answer in order:
1. text_in_box: the text you can read inside the red rectangle, verbatim, or
   "" if there is none.
2. element_in_box: what control the rectangle actually encloses, in a few
   words.
3. verdict: "correct" if the rectangle tightly encloses exactly the control the
   instruction names; "loose" if it contains that control but also much more;
   "partial" if it cuts off part of that control or includes a neighbour;
   "wrong_element" if it encloses a different control or empty space.
4. reason: one sentence.
5. corrected_bbox: if the verdict is not "correct", the tight box of the
   control the instruction names as [x1, y1, x2, y2] {coords}
   IMAGE 2; null if the verdict is "correct" or the control is not visible in
   image 2.

Return exactly one JSON object with those five keys."""

REFINE_GROUND_USER = """This is a magnified region of a screenshot of {app}.
Locate the element this instruction refers to: "{instruction}"
{previous} {target}
Output only its bounding box as JSON:
[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
{coords} this image. If the control is not in this image,
output []."""

REFINE_WRONG_SENTENCE = 'A previous attempt boxed "{text_in_box}" instead, which is wrong.'
REFINE_LOOSE_SENTENCE = "A previous attempt drew a box that was too loose around it."
REFINE_UNKNOWN_SENTENCE = "A previous attempt boxed a different control, which is wrong."
REFINE_LABEL_CLAUSE = 'The control you are looking for is labelled "{label}"{icon_clause}.'
REFINE_NO_LABEL_CLAUSE = "The control you are looking for has no text label{icon_clause}."
REFINE_ICON_CLAUSE = " and carries a {icon} icon"

CALIBRATE_USER = "Describe this screenshot in one word."

# The unit clause every coordinate sentence hangs on, rendered per CoordSpace by coords_clause().
COORDS_THOUSAND = "on a 0-1000 scale of"
COORDS_PIXELS = "in pixel coordinates ({w}x{h} pixels) of"


def render(template: str, **fields) -> str:
    """Fill {name} placeholders for the given names only. str.format is not used because
    the templates contain JSON skeletons full of braces."""
    out = template
    for name, value in fields.items():
        out = out.replace("{" + name + "}", str(value))
    return out


def coords_clause(space) -> str:
    """What replaces {coords}: the 0-1000 scale, or the pixel size of the image as sent."""
    if space.mode == "thousand":
        return COORDS_THOUSAND
    w, h = space.shown_size
    return render(COORDS_PIXELS, w=int(w), h=int(h))


def prompt_hash(coord_mode="thousand") -> str:
    """SHA-256 over every UPPER_CASE constant of this module, in name order, plus the coordinate
    mode the templates are rendered in."""
    mod = sys.modules[__name__]
    parts = [f"{name}={getattr(mod, name)!r}" for name in sorted(dir(mod)) if name.isupper()]
    parts.append(f"coord_mode={coord_mode}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
