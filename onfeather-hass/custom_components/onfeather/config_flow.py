"""Config flow for the OnFeather integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_URL,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from . import OnFeatherConfigEntry
from .api import OnFeatherAuthError, OnFeatherClient, OnFeatherError
from .const import (
    CONF_MAX_HISTORY,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_URL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER = vol.Schema(
    {
        vol.Required(CONF_URL, default=DEFAULT_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)

STEP_REAUTH = vol.Schema(
    {
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        )
    }
)


async def _reachable(hass: HomeAssistant, url: str, api_key: str | None) -> OnFeatherClient:
    """Prove the router answers before an entry is written."""
    client = OnFeatherClient(async_get_clientsession(hass), url, api_key)
    await client.status()
    return client


class OnFeatherConfigFlow(ConfigFlow, domain=DOMAIN):
    """Add one `of-free` server."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask where the router is."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                client = await _reachable(
                    self.hass, user_input[CONF_URL], user_input.get(CONF_API_KEY)
                )
            except OnFeatherAuthError:
                errors["base"] = "invalid_auth"
            except OnFeatherError:
                errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Unexpected error reaching OnFeather")
                errors["base"] = "unknown"
            else:
                await self.async_set_unique_id(client.base_url)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="OnFeather",
                    data={
                        CONF_URL: client.base_url,
                        CONF_API_KEY: user_input.get(CONF_API_KEY),
                    },
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER, errors=errors
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """The router started asking for a key, or the key changed."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the key again."""
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await _reachable(
                    self.hass, entry.data[CONF_URL], user_input.get(CONF_API_KEY)
                )
            except OnFeatherAuthError:
                errors["base"] = "invalid_auth"
            except OnFeatherError:
                errors["base"] = "cannot_connect"
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_API_KEY: user_input.get(CONF_API_KEY)}
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_REAUTH,
            errors=errors,
            description_placeholders={"url": entry.data[CONF_URL]},
        )

    @staticmethod
    @callback
    def async_get_options_flow(entry: OnFeatherConfigEntry) -> OnFeatherOptionsFlow:
        """Get the options flow."""
        return OnFeatherOptionsFlow()


class OnFeatherOptionsFlow(OptionsFlow):
    """How the conversation entity should behave."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure the agent."""
        if user_input is not None:
            # 20260804 ++ RG #HASS An empty selection means no control, and must not persist as [].
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                await self._schema(), self.config_entry.options
            ),
        )

    async def _schema(self) -> vol.Schema:
        return vol.Schema(
            {
                vol.Required(CONF_MODEL, default=DEFAULT_MODEL): SelectSelector(
                    SelectSelectorConfig(options=await self._models(), custom_value=True)
                ),
                vol.Optional(CONF_PROMPT): TemplateSelector(),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(label=api.name, value=api.id)
                            for api in llm.async_get_apis(self.hass)
                        ],
                        multiple=True,
                    )
                ),
                vol.Optional(
                    CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY
                ): NumberSelector(
                    NumberSelectorConfig(min=0, max=200, step=1, mode=NumberSelectorMode.BOX)
                ),
            }
        )

    async def _models(self) -> list[SelectOptionDict]:
        """Offer what the router can reach, without insisting on it.

        `auto` is the point of the product, so it leads. The rest are there for
        the times you want one specific model, and `custom_value` covers the
        aliases the router understands but does not advertise, such as
        `private`.
        """
        options = [SelectOptionDict(label="auto — let the router choose", value="auto")]

        coordinator = getattr(self.config_entry, "runtime_data", None)
        if coordinator is None:
            return options

        try:
            listed = await coordinator.client.models()
        except OnFeatherError:
            # 20260804 ++ RG #HASS An unreachable router must not block the options form.
            return options

        options.extend(
            SelectOptionDict(label=model["id"], value=model["id"])
            for model in listed
            if model.get("id") and model["id"] != "auto"
        )
        return options
