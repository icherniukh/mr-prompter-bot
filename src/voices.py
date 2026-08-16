from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceOption:
    id: str
    name: str
    gender: str
    flag: str
    country: str
    desc: str

    @property
    def label(self) -> str:
        gender_icon = "👨" if self.gender == "Male" else "👩"
        return f"{self.flag} {self.name} ({gender_icon} {self.desc})"


# Curated list of premier neural voices
CURATED_VOICES: list[VoiceOption] = [
    # 🇺🇦 Ukraine (2 native Ukrainian voices)
    VoiceOption("uk-UA-OstapNeural", "Остап", "Male", "🇺🇦", "UA", "Ukrainian Male"),
    VoiceOption("uk-UA-PolinaNeural", "Поліна", "Female", "🇺🇦", "UA", "Ukrainian Female"),


    # 🇺🇸 United States (12 voices: 6 Male, 6 Female)
    VoiceOption("en-US-ChristopherNeural", "Christopher", "Male", "🇺🇸", "US", "Deep & Authoritative"),
    VoiceOption("en-US-GuyNeural", "Guy", "Male", "🇺🇸", "US", "Casual & Relaxed"),
    VoiceOption("en-US-JennyNeural", "Jenny", "Female", "🇺🇸", "US", "Natural & Friendly"),

    VoiceOption("en-US-AriaNeural", "Aria", "Female", "🇺🇸", "US", "Expressive Narrator"),
    VoiceOption("en-US-EricNeural", "Eric", "Male", "🇺🇸", "US", "Young & Energetic"),
    VoiceOption("en-US-AnaNeural", "Ana", "Female", "🇺🇸", "US", "Soft & Gentle"),
    VoiceOption("en-US-BrianNeural", "Brian", "Male", "🇺🇸", "US", "Dynamic & Confident"),
    VoiceOption("en-US-AvaNeural", "Ava", "Female", "🇺🇸", "US", "Warm & Clear"),
    VoiceOption("en-US-RogerNeural", "Roger", "Male", "🇺🇸", "US", "Mature & Rich"),
    VoiceOption("en-US-EmmaNeural", "Emma", "Female", "🇺🇸", "US", "Crisp & Modern"),
    VoiceOption("en-US-SteffanNeural", "Steffan", "Male", "🇺🇸", "US", "Formal & Direct"),
    VoiceOption("en-US-MichelleNeural", "Michelle", "Female", "🇺🇸", "US", "Professional"),

    # 🇬🇧 United Kingdom (6 voices: 3 Male, 3 Female)
    VoiceOption("en-GB-RyanNeural", "Ryan", "Male", "🇬🇧", "UK", "Broadcast & Formal"),
    VoiceOption("en-GB-SoniaNeural", "Sonia", "Female", "🇬🇧", "UK", "Refined & Warm"),
    VoiceOption("en-GB-ThomasNeural", "Thomas", "Male", "🇬🇧", "UK", "Classic British"),
    VoiceOption("en-GB-LibbyNeural", "Libby", "Female", "🇬🇧", "UK", "Crisp & Articulate"),
    VoiceOption("en-GB-MaisieNeural", "Maisie", "Female", "🇬🇧", "UK", "Youthful & Bright"),
    VoiceOption("en-US-AndrewNeural", "Andrew", "Male", "🇺🇸", "US", "Warm & Conversational"),


    # 🇦🇺 Australia (2 voices)
    VoiceOption("en-AU-WilliamMultilingualNeural", "William", "Male", "🇦🇺", "AU", "Australian Classic"),
    VoiceOption("en-AU-NatashaNeural", "Natasha", "Female", "🇦🇺", "AU", "Australian Warm"),

    # 🇨🇦 Canada (2 voices)
    VoiceOption("en-CA-LiamNeural", "Liam", "Male", "🇨🇦", "CA", "Canadian Male"),
    VoiceOption("en-CA-ClaraNeural", "Clara", "Female", "🇨🇦", "CA", "Canadian Female"),

    # 🇮🇪 Ireland (2 voices)
    VoiceOption("en-IE-ConnorNeural", "Connor", "Male", "🇮🇪", "IE", "Irish Male"),
    VoiceOption("en-IE-EmilyNeural", "Emily", "Female", "🇮🇪", "IE", "Irish Female"),

    # 🇳🇿 New Zealand (2 voices)
    VoiceOption("en-NZ-MitchellNeural", "Mitchell", "Male", "🇳🇿", "NZ", "Kiwi Male"),
    VoiceOption("en-NZ-MollyNeural", "Molly", "Female", "🇳🇿", "NZ", "Kiwi Female"),

    # 🇮🇳 India (2 voices)
    VoiceOption("en-IN-PrabhatNeural", "Prabhat", "Male", "🇮🇳", "IN", "Indian Male"),
    VoiceOption("en-IN-NeerjaExpressiveNeural", "Neerja", "Female", "🇮🇳", "IN", "Indian Expressive"),

    # 🇿🇦 South Africa (2 voices)
    VoiceOption("en-ZA-LukeNeural", "Luke", "Male", "🇿🇦", "ZA", "South African Male"),
    VoiceOption("en-ZA-LeahNeural", "Leah", "Female", "🇿🇦", "ZA", "South African Female"),
]

VOICES_BY_ID = {v.id: v for v in CURATED_VOICES}


def get_voice_by_id(voice_id: str | None) -> VoiceOption | None:
    if not voice_id:
        return None
    return VOICES_BY_ID.get(voice_id)
