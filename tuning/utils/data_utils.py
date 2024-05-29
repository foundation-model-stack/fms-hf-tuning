# Standard
import re


def apply_custom_formatting_template(dataset, template, eos_token=""):
    """Function to format datasets with Alpaca style / other templates.
    Args:
        dataset: the HF Dataset element loaded from a JSON or DatasetDict object.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}
        eos_token: string EOS token to be appended while formatting data to a single sequence.
            Defaults to empty
    Returns:
        Formatted HF Dataset, dataset_field name that contains formatted data.
    """

    formatted_dataset_field = "formatted_data_field"
    template += eos_token

    def formatter(element):
        def replace_text(match_obj):
            captured_groups = match_obj.groups()
            if len(captured_groups) != 1:
                raise ValueError(
                    "Unexpectedly captured multiple groups in template formatting"
                )

            index_object = captured_groups[0]
            if index_object not in element:
                raise KeyError("Requested template string is not a valid key in dict")

            return element[index_object]

        return {
            formatted_dataset_field: re.sub(
                r"{{([\s0-9a-zA-Z_\-\.]+)}}", replace_text, template
            )
        }

    return dataset.map(formatter), formatted_dataset_field
