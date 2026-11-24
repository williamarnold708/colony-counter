# Privacy Policy — Colony Counter (Android)

Last updated: 2026-09-23

Colony Counter is an Android app that counts microbial colonies in a photo of an
agar plate. **All detection runs on your device.** The app does not have a
server to talk to, does not use the internet, and does not collect, transmit,
or sell any data, to us or to anyone else.

## What the app does with a photo

When you take or choose a plate photo, it is analysed by a detection model
bundled inside the app, on your device. The photo is never uploaded anywhere.

If you use the **save for training** feature to record a corrected count, the
photo and the marks you made are written to the app's own storage on your
device (`Android/data/com.agarcount.app/files/`). That stays on your
device — the app has no way to send it elsewhere — unless you copy it off
yourself, for example over a USB cable.

## Permissions

- **Camera / photo library** — to let you take or choose the plate photo you
  want analysed. Used only when you tap to add a photo.
- **Internet** — Android requires this permission for the app to open any
  network socket, including a purely local one. The app uses it only to let
  its own interface (a web page) talk to its own detection code, both running
  on your device at the fixed address 127.0.0.1. No connection is made to
  any other address.

## What we don't do

No accounts, no analytics, no advertising, no crash-reporting service, no
third-party SDKs that collect data. Nothing about you or your photos is
visible to us — we have no server that could see it.

## Children's privacy

The app is a laboratory tool and is not directed at children. It does not
knowingly collect data from anyone, of any age, because it does not collect
data from anyone at all.

## Changes to this policy

Any change to this policy will be made at this same page, in the project's
public repository.

## Contact

[colony-counter on GitHub](https://github.com/williamarnold708/colony-counter) — open an issue, or reach the developer at williamarnold708@gmail.com.
