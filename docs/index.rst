:layout: landing

ASQI Engineer
=============

**AI Systems Quality Index Engineer** - A comprehensive framework for systematic testing and quality assurance of AI systems. Built for reliability and scale, ASQI enables rigorous evaluation through containerized test packages, automated assessment, and durable execution workflows.

.. container:: buttons

    `Quick Start <quickstart.html>`_
    `GitHub <https://github.com/asqi-engineer/asqi-engineer>`_

Key Features
------------

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: ⚡ Durable Execution
        :text-align: center

        DBOS-powered fault tolerance with automatic retry and recovery for reliable test execution.

    .. grid-item-card:: 🐳 Container Isolation
        :text-align: center

        Reproducible testing in isolated Docker environments with consistent, repeatable results.

    .. grid-item-card:: 🎭 Multi-System Orchestration
        :text-align: center

        Coordinate target, simulator, and evaluator systems in complex testing workflows.

    .. grid-item-card:: 📊 Flexible Assessment
        :text-align: center

        Configurable score cards map technical metrics to business-relevant outcomes.

    .. grid-item-card:: 🛡️ Type-Safe Configuration
        :text-align: center

        Pydantic schemas with JSON Schema generation provide IDE integration and validation.

    .. grid-item-card:: 🔄 Modular Workflows
        :text-align: center

        Separate validation, test execution, and evaluation phases for flexible CI/CD integration.


Test Packages
-------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: 🔒 Security Testing
        :text-align: center

        Comprehensive vulnerability assessment with **Garak** (40+ security probes) and **DeepTeam** (advanced red teaming) frameworks.

        +++

        .. button-ref:: llm-test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Security Containers

    .. grid-item-card:: 💬 Conversation Quality  
        :text-align: center

        Multi-turn dialogue testing with **persona-based simulation** and **LLM-as-judge evaluation** for realistic chatbot assessment.

        +++

        .. button-ref:: llm-test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Quality Testing

    .. grid-item-card:: 🎯 Trustworthiness
        :text-align: center

        Academic-grade evaluation across **6 trust dimensions** using the **TrustLLM** framework for comprehensive assessment.

        +++

        .. button-ref:: llm-test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Trust Evaluation

    .. grid-item-card:: 🔧 Custom Testing
        :text-align: center

        Build **domain-specific test containers** with standardized interfaces and **multi-system orchestration** capabilities.

        +++

        .. button-ref:: custom-test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Create Containers

Contributors
------------

.. contributors:: lepture/shibuya
    :avatars:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   :hidden:

   quickstart
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Configuration:
   :hidden:

   configuration
   llm-test-containers
   custom-test-containers

.. toctree::
   :maxdepth: 2
   :caption: Reference:
   :hidden:

   cli
   examples
   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Links:
   :hidden:

   GitHub Repository <https://github.com/asqi-engineer/asqi-engineer>
   Discussions <https://github.com/asqi-engineer/asqi-engineer/discussions>

