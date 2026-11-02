# Tracker data

This folder holds the task list behind the team's Google Sheets tracker for this project. The sheet reads `tasks.csv` from this branch, so a change here shows up in the sheet within about an hour.

* **Status:** Done, In progress, Not started, Blocked or Deferred.
* **Milestone:** Tue 1 pm (3 Nov 2026, 1 p.m. IST), Wed 9:30 (4 Nov 2026, 9:30 a.m. IST), or Later (outside this sprint).
* **Start and Finish:** planned time slots in IST. For finished tasks they are the actual times, rounded.
* **Build (h):** estimated agent working time. **Compute wait (h):** estimated background compute, such as simulations and pipeline runs.
* **Evidence:** the commit, file or pull request that shows the work.

This branch only carries the tracker. It is not meant to be merged into `main`.
