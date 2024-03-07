# Contributing

👍🎉 First off, thank you for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## What Should I Know Before I Get Started?

### Code of Conduct

This project adheres to the [Contributor Covenant](./code-of-conduct.md). By participating, you are expected to uphold this code.

Please report unacceptable behavior to one of the [Code Owners](./CODEOWNERS).

### How Do I Start Contributing?

The below workflow is designed to help you begin your first contribution journey. It will guide you through creating and picking up issues, working through them, having your work reviewed, and then merging.

Help on open source projects is always welcome and there is always something that can be improved. For example, documentation (like the text you are reading now) can always use improvement, code can always be clarified, variables or functions can always be renamed or commented on, and there is always a need for more test coverage. If you see something that you think should be fixed, take ownership! Here is how you get started:

## How Can I Contribute?
TODO: Add link to ADR  and add template to this repository

For any contributions that need design changes/API changes, first contribute an ADR. Reason for ADR: teams agree on the design, to avoid back and forth after writing code. An ADR gives context on the code being written.

When contributing, it's useful to start by looking at [issues](https://github.com/foundation-model-stack/fms-hf-tuning/issues). After picking up an issue, writing code, or updating a document, make a pull request and your work will be reviewed and merged. If you're adding a new feature or find a bug, it's best to [write an issue](https://github.com/foundation-model-stack/fms-hf-tuning/issues/new) first to discuss it with maintainers. 

To contribute to this repo, you'll use the Fork and Pull model common in many open source repositories. For details on this process, check out [The GitHub Workflow
Guide](https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md)
from Kubernetes.

When your contribution is ready, you can create a pull request. Pull requests are often referred to as "PR". In general, we follow the standard [GitHub pull request](https://help.github.com/en/articles/about-pull-requests) process. Follow the template to provide details about your pull request to the maintainers. It's best to break your contribution into smaller PRs with incremental changes, and include a good description of the changes.

Before sending pull requests, make sure your changes pass formatting, linting and unit tests.

#### Code Review

Once you've [created a pull request](#how-can-i-contribute), maintainers will review your code and may make suggestions to fix before merging. It will be easier for your pull request to receive reviews if you consider the criteria the reviewers follow while working. Remember to:

- Run tests locally and ensure they pass
- Follow the project coding conventions
- Write detailed commit messages
- Break large changes into a logical series of smaller patches, which are easy to understand individually and combine to solve a broader issue

Maintainers will perform "squash and merge" actions on PRs in this repo, so it doesn't matter how many commits your PR has, as they will end up being a single commit after merging.

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report ✏️, reproduce the behavior 💻, and find related reports 🔎.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues using the Bug Report template]((https://github.com/foundation-model-stack/fms-hf-tuning/issues/new?template=bug_report.md). Create an issue on that and provide the information suggested in the bug report issue template. 

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features, tools, and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion ✏️ and find related suggestions 🔎

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues using the Feature Request template](https://github.com/foundation-model-stack/fms-hf-tuning/issues/new?template=feature_request.md). Create an issue and provide the information suggested in the feature requests or user story issue template.

#### How Do I Submit A (Good) Improvement Item?

Improvements to existing functionality are tracked as [GitHub issues using the User Story template](https://github.com/foundation-model-stack/fms-hf-tuning/issues/new?template=user_story.md). Create an issue and provide the information suggested in the feature requests or user story issue template.

## Development

### Set up your dev environment

The following tools are required:

- [git](https://git-scm.com)
- [python](https://www.python.org) (v3.8+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

Installation:
``` 
pip install -r requirements.txt
pip install -U datasets
pip install -e .
```

### Unit tests

Work in process, to be completed soon. 

## Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these issues:

- Issues with the [`good first issue` label](https://github.com/foundation-model-stack/fms-hf-tuning/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - these should only require a few lines of code and are good targets if you're just starting contributing.
- Issues with the [`help wanted` label](https://github.com/foundation-model-stack/fms-hf-tuning/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) - these range from simple to more complex, but are generally things we want but can't get to in a short time frame.
