# Contributing to Jespipe

Do you have something that you wish to contribute to Jespipe? **Here is how can help!**

Please take a moment to review this document so that the contribution process will be easy and effective for everyone involved.

Following these guidelines helps you communicate that you respect the developer managing and developing Jespipe. In return, they should reciprocate that respect while they are addressing your issue or assesing your submitted patches and features.

### Table of Contents

* [Using the Issue Tracker](#using-the-issue-tracker)
* [Issues and Labels](#issues-and-labels)
* [Bug Reports](#bug-reports)
* [Feature Requests](#feature-requests)
* [Pull Requests](#pull-requests)
* [Discussions](#discussions)
* [Code Guidelines](#code-guidelines)
* [License](#license)

## Using the Issue Tracker
The issue tracker is the preferred way for tracking [bug reports](#bub-reports), [feature requests](#feature-requests), and [submitted pull requests](#pull-requests), but please follow these guidelines for the issue tracker:

* Please **do not** use the issue tracker for personal issues and/or support requests. The [Discussions](#discussions) page is a better place to get help for personal support requests.

* Please **do not** derail or troll issues. Keep the discussion on track and have respect for the other users/contributors of Jespipe.

* Please **do not** post comments consisting soley of "+1", ":thumbsup:", or something similiar. Use [GitHub's "reactions" feature](https://blog.github.com/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) instead.
  * The maintainers of Jespipe reserve the right to delete comments that violate this rule.

* Please **do not** repost or reopen issues that have been marked as `resolved`. Please either submit a new issue or browser through previous issues.
  * The maintainers of Jespipe reserve the right to delete issues that violate this rule.

## Issues and Labels

The Jespipe issue tracker uses a variety of labels to help organize and identify issues. Here is a list of some of these labels, and how the maintainers of Jespipe use them:

* `bug` - Issues reported in the Jespipe source code that either produce errors or unexpected behavior. Bug fixes require a mini version bump (ex. `v0.0.1` to `v0.0.2`)

* `confirmed` - Issues marked `bug` that have be confirmed to be reproducible on a separate system.

* `documentation` - Issues for improving or updating Jespipe's documentation.

* `examples` - Issues involving the examples included with Jespipe.

* `build` - Issues with Jespipe's build system, installation, or unit tests.

* `refactor` - Issues that pertain to improvements in Jespipe's source code.

* `meta` - Issues with Jespipe itself or the Jespipe GitHub repository.

* `feature request` - Issues asking for a new feature to be, or an existing feature to be extended or modified. New features require a minor version bump (ex. `v0.0.1` to `v0.1.1`).

* `enchancement` - Issues marked `feature request` that will be included in the next release of Jespipe.

* `help wanted` - Issues where we need help from the community to solve.

For a complete look at Jespipe's labels, see our [project labels page](https://github.com/NucciTheBoss/jespipe/labels)! 

## Bug Reports

A bug is a *demonstrable problem* that is caused by the code in the repository. Good bug reports make Jespipe more robust, so thank you for taking the time to report issues in the source code!

Guidelines for reporting bugs in Jespipe:

1. **Validate your plugins** &mdash; ensure that your issue is not being caused by either a semantic or syntatic error in your plugin's code.

2. **Use the GitHub issue search** &mdash; check if the issue you are encountering has already been reported.

3. **Check if the issue has already been fixed** &mdash; try to reproduce your issue using the latest `main` in the Jespipe repository.

4. **Isolate the problem** &mdash; the more pin-pointed the issue is, the easier time the Jespipe developers will have fixing it.

A good bug report shouldn't leave others needing to chase you up for more information. Please try to be as detailed as possible in your report. What is your environment? What steps will reproduce the issue? What Operating System are you experiencing the problem on? Have you had the same results on a different Operating System? What would you expect to be the outcome? All these details will help the developers fix any potential bugs.

Example Bug Report:

> Short and desciptive bug report title
> 
> ### System Info (please complete the following information):
>  - OS: [e.g. RedHat Enterprise Linux 8]
>  - Jespipe Version: [e.g. v0.0.1]
>  - GCC Version: [e.g. 8.3.1] 
>  - Python Version: [e.g. 3.9.5]
>  - OpenMPI Version: [e.g. 4.0.5]
> 
> ### Describe the Bug:
> A clear and concise description of what the bug is.
> 
> ### Copy of Stacktrace:
> ```
> Copy of stack printed out to command line or logfile.
> ```
> 
> ### Steps to Reproduce:
> Steps to reproduce the behavior:
> 1. Commands used
> 2. Plugins called
> 3. Stage where error occurred
> 4. See error
> 
> ### Expected Behavior:
> A clear and concise description of what you expected to happen.
> 
> ### Screenshots:
> If applicable, add screenshots to help explain your problem.
> 
> #### Additional context:
> Add any other context about the problem here.

You can use the [Bug Report template](https://github.com/NucciTheBoss/jespipe/issues/new?assignees=NucciTheBoss&labels=bug&template=bug-report.md&title=BUG%3A+Error+encountered+during+Jespipe+runtime) to submit issues to the Jespipe developers.

## Feature Requests

Feature requests to improve/extend are welcome, but please take a moment to determine whether your idea with the scope and aim of Jespipe. It is up to *you* to make a strong case to convince Jespipe's developers of the merits of adding this feature to Jespipe. Please provide as much detail and context as possible.

Example Feature Request:

> Short and descriptive feature request title
> 
> ### Is your feature request related to a problem? Please describe:
> A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]; No, but I would love if Jespipe could do [...]; No, but I think this process could be improved; etc.
> 
> ### Describe the solution you'd like:
> A clear and concise description of what you want to happen. Ex. If I enter  this command, this should happen [...].
> 
> ### Describe alternatives you've considered:
> A clear and concise description of any alternative solutions or features > you've considered.
> 
> #### Additional context:
> Add any other context or screenshots about the feature request here.

You can use the [Feature Request template](https://github.com/NucciTheBoss/jespipe/issues/new?assignees=NucciTheBoss&labels=enhancement%2C+question&template=feature-request.md&title=FEATURE+REQUEST%3A+Potential+improvement+to+Jespipe) to submit feature requests to the Jespipe developers.

## Pull Requests

Good pull requests&mdash;patches, improvements, new features&mdash;are a huge help. These pull requests should remain focused in scope and should not contain unrelated commits.

**Ask first** before embarking on any **significant** pull request (eg. implementing features, refactoring code, porting to a different language), otherwise you risk spending a lot of time working on something that Jespipe's developers might not want to merge into the project! For trivial things, or things that don't require a lot of your time, you can go ahead and make a pull request.

Adhering to the following process is the best way to get your work
included in the project:

1. [Fork](https://help.github.com/articles/fork-a-repo/) the project, clone your fork,
   and configure the remotes:

   ```bash
   # Clone your fork of the repo into the current directory
   git clone https://github.com/<your-username>/jespipe.git
   # Navigate to the newly cloned directory
   cd bootstrap
   # Assign the original repo to a remote called "upstream"
   git remote add upstream https://github.com/NucciTheBoss/jespipe.git
   ```

2. If you cloned a while ago, get the latest changes from upstream:

   ```bash
   git checkout main
   git pull upstream main
   ```

3. Create a new topic branch (off the main project development branch) to
   contain your feature, change, or fix:

   ```bash
   git checkout -b <topic-branch-name>
   ```

4. Commit your changes in logical chunks. Please adhere to these [git commit
   message guidelines](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
   or your code is unlikely be merged into the main project. Use Git's
   [interactive rebase](https://help.github.com/articles/about-git-rebase/)
   feature to tidy up your commits before making them public.

5. Locally merge (or rebase) the upstream development branch into your topic branch:

   ```bash
   git pull [--rebase] upstream main
   ```

6. Push your topic branch up to your fork:

   ```bash
   git push origin <topic-branch-name>
   ```

7. [Open a Pull Request](https://help.github.com/articles/about-pull-requests/)
    with a clear title and description against the `main` branch.

**IMPORTANT**: By submitting a patch, improvement, or new feature, you agree to allow the owners of Jespipe to license your work under the terms of the [GNU General Public License version 3](../LICENSE) if it includes code changes and under the terms of the [BSD 3-Clause License](https://creativecommons.org/licenses/by/3.0/) if it includes documentation changes.

## Discussions

GitHub discussions are a great place to connect with other users of Jespipe as well as discuss potential features and resolve personal support questions. It is expected that the users of Jespipe remain respectful of each other. Discussion moderators reserve the right the suspend discussions and/or delete posts that do not follow this rule.

## Code Guidelines

### Python

* Adhere to the Python code style guidelines outlined in [Python Enhancement Proposal 8](https://pep8.org/).

* Adhere to the Python docstring conventions outlined in [Python Enhancement Proposal 257](https://www.python.org/dev/peps/pep-0257/).
  * *Jespipe docstrings are written in [reStructuredText Markup](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)*.
  
* Adhere to the Python Type Hint guidelines outlined in [Python Enhancement Proposal 484](https://www.python.org/dev/peps/pep-0484/)

## License

By contributing your code to Jespipe, you agree to license your contribution under the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.en.html). By contributing to the Jespipe documentation, you agree to license your contribution under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).